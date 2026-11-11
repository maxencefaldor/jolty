use serde::Deserialize;

use crate::junction::{
    Depletion, THERMAL_VOLTAGE, below_one, critical_voltage, depletion_charge, junction_current,
    limit_junction,
};
use crate::stamp::Stamp;
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, Output, ParameterError,
    Quantity,
};

/// The doping order of a bipolar transistor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Polarity {
    /// An NPN transistor, conducting with its base above its emitter.
    Npn,
    /// A PNP transistor, the NPN's mirror: every voltage and current reversed.
    Pnp,
}

impl Polarity {
    /// `1` for NPN, `−1` for PNP: the factor that orients voltages and currents as an
    /// NPN's.
    #[must_use]
    pub const fn sign(self) -> f64 {
        match self {
            Self::Npn => 1.0,
            Self::Pnp => -1.0,
        }
    }
}

/// How a bipolar transistor is built, which sets where its junction to the substrate lies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Construction {
    /// Built vertically, with the junction to the substrate at the collector.
    Vertical,
    /// Built laterally, with the junction to the substrate at the base.
    Lateral,
}

/// A bipolar junction transistor, in the Gummel-Poon model of SPICE, as ngspice implements
/// it.
///
/// Two junctions, base-emitter and base-collector, each carry an ideal
/// [junction current](crate::junction) and a leakage one. The transport current from
/// collector to emitter, `(Ibe − Ibc) / qb`, falls with the base charge `qb`, which models
/// the Early effect and high injection. The base resistance falls from `rb` towards `rbm`
/// as the base charge or the base current grows. Each junction holds a depletion charge,
/// and a diffusion charge set by its transit time; the forward transit time grows with
/// the current and the base-collector voltage. Part of the base-collector depletion charge
/// sits on the external base. A base, collector or emitter resistance adds an internal
/// node, labelled `base`, `collector` or `emitter`.
///
/// A transistor with a substrate has a fourth terminal, `substrate`, joined by a junction
/// to its internal collector if it is vertical or to its internal base if it is lateral.
/// The junction conducts the minimum conductance, and its diode's current if it has a
/// saturation current; its depletion capacitance grows linearly in forward bias.
#[derive(Debug)]
pub struct BipolarTransistor {
    stamp: Stamp<6>,
    sign: f64,
    collector_conductance: f64,
    emitter_conductance: f64,
    base: Option<BaseResistance>,
    junctions: Junctions,
    charges: Option<Charges>,
    substrate: Option<Substrate>,
    critical_voltage: f64,
    /// The base-emitter and base-collector voltages of the last evaluation, from which the
    /// next steps are limited.
    last: [f64; 2],
}

/// Local node indices: the terminals, then the internal collector, base and emitter, which
/// are the terminals themselves without series resistance.
const COLLECTOR: usize = 0;
const BASE: usize = 1;
const EMITTER: usize = 2;
const INNER_COLLECTOR: usize = 3;
const INNER_BASE: usize = 4;
const INNER_EMITTER: usize = 5;

/// The parameters of a [`BipolarTransistor`], with SPICE's defaults. A parameter SPICE
/// treats as infinite at zero, such as an Early voltage, is infinite when left out.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BipolarTransistorParameters {
    /// The doping order; default `npn`.
    pub polarity: Polarity,
    /// The transport saturation current, `IS`, in amperes; default 1e-16.
    pub saturation_current: f64,
    /// The ideal maximum forward current gain, `BF`; default 100.
    pub forward_beta: f64,
    /// The ideal maximum reverse current gain, `BR`; default 1.
    pub reverse_beta: f64,
    /// The forward emission coefficient, `NF`; default 1.
    pub forward_emission_coefficient: f64,
    /// The reverse emission coefficient, `NR`; default 1.
    pub reverse_emission_coefficient: f64,
    /// The forward Early voltage, `VAF`, in volts; infinite by default.
    pub forward_early_voltage: Option<f64>,
    /// The reverse Early voltage, `VAR`, in volts; infinite by default.
    pub reverse_early_voltage: Option<f64>,
    /// The forward knee current, `IKF`, where high injection sets in, in amperes; infinite
    /// by default.
    pub forward_knee_current: Option<f64>,
    /// The reverse knee current, `IKR`, in amperes; infinite by default.
    pub reverse_knee_current: Option<f64>,
    /// The base-emitter leakage saturation current, `ISE`, in amperes; default 0.
    pub base_emitter_leakage_current: f64,
    /// The base-emitter leakage emission coefficient, `NE`; default 1.5.
    pub base_emitter_leakage_emission_coefficient: f64,
    /// The base-collector leakage saturation current, `ISC`, in amperes; default 0.
    pub base_collector_leakage_current: f64,
    /// The base-collector leakage emission coefficient, `NC`; default 2.
    pub base_collector_leakage_emission_coefficient: f64,
    /// The zero-bias base resistance, `RB`, in ohms; default 0.
    pub base_resistance: f64,
    /// The base resistance at high current, `RBM`, in ohms; the base resistance by default.
    pub minimum_base_resistance: Option<f64>,
    /// The base current, `IRB`, at which the base resistance falls halfway to its minimum,
    /// in amperes; without it, the base resistance falls with the base charge.
    pub base_resistance_current: Option<f64>,
    /// The emitter resistance, `RE`, in ohms; default 0.
    pub emitter_resistance: f64,
    /// The collector resistance, `RC`, in ohms; default 0.
    pub collector_resistance: f64,
    /// The zero-bias base-emitter depletion capacitance, `CJE`, in farads; default 0.
    pub base_emitter_capacitance: f64,
    /// The base-emitter built-in potential, `VJE`, in volts; default 0.75.
    pub base_emitter_potential: f64,
    /// The base-emitter grading coefficient, `MJE`, below 1; default 0.33.
    pub base_emitter_grading: f64,
    /// The ideal forward transit time, `TF`, in seconds; default 0.
    pub forward_transit_time: f64,
    /// How much the forward transit time grows with bias, `XTF`; default 0.
    pub transit_time_bias_coefficient: f64,
    /// The base-collector voltage, `VTF`, that sets how the forward transit time grows with
    /// it, in volts; infinite by default.
    pub transit_time_voltage: Option<f64>,
    /// The current, `ITF`, above which the forward transit time grows, in amperes;
    /// default 0.
    pub transit_time_current: f64,
    /// The zero-bias base-collector depletion capacitance, `CJC`, in farads; default 0.
    pub base_collector_capacitance: f64,
    /// The base-collector built-in potential, `VJC`, in volts; default 0.75.
    pub base_collector_potential: f64,
    /// The base-collector grading coefficient, `MJC`, below 1; default 0.33.
    pub base_collector_grading: f64,
    /// The fraction of the base-collector capacitance on the internal base, `XCJC`, the
    /// rest on the external base; default 1.
    pub internal_base_fraction: f64,
    /// The ideal reverse transit time, `TR`, in seconds; default 0.
    pub reverse_transit_time: f64,
    /// The fraction of each junction's potential above which its depletion capacitance is
    /// extended linearly, `FC`, below 1; default 0.5.
    pub depletion_coefficient: f64,
    /// How the transistor is built, if it has a substrate, as ngspice's `SUBS` of 1 and −1
    /// builds it; without it, the transistor has none. The parameters below, of the
    /// junction to the substrate, need one.
    pub substrate: Option<Construction>,
    /// The substrate junction's saturation current, `ISS`, in amperes; without it, the
    /// junction carries the minimum conductance only.
    pub substrate_saturation_current: Option<f64>,
    /// The substrate junction's emission coefficient, `NS`; default 1.
    pub substrate_emission_coefficient: Option<f64>,
    /// The substrate junction's zero-bias depletion capacitance, `CJS`, in farads;
    /// default 0.
    pub substrate_capacitance: Option<f64>,
    /// The substrate junction's built-in potential, `VJS`, in volts; default 0.75.
    pub substrate_potential: Option<f64>,
    /// The substrate junction's grading coefficient, `MJS`, below 1; default 0.
    pub substrate_grading: Option<f64>,
}

impl Default for BipolarTransistorParameters {
    fn default() -> Self {
        Self {
            polarity: Polarity::Npn,
            saturation_current: 1e-16,
            forward_beta: 100.0,
            reverse_beta: 1.0,
            forward_emission_coefficient: 1.0,
            reverse_emission_coefficient: 1.0,
            forward_early_voltage: None,
            reverse_early_voltage: None,
            forward_knee_current: None,
            reverse_knee_current: None,
            base_emitter_leakage_current: 0.0,
            base_emitter_leakage_emission_coefficient: 1.5,
            base_collector_leakage_current: 0.0,
            base_collector_leakage_emission_coefficient: 2.0,
            base_resistance: 0.0,
            minimum_base_resistance: None,
            base_resistance_current: None,
            emitter_resistance: 0.0,
            collector_resistance: 0.0,
            base_emitter_capacitance: 0.0,
            base_emitter_potential: 0.75,
            base_emitter_grading: 0.33,
            forward_transit_time: 0.0,
            transit_time_bias_coefficient: 0.0,
            transit_time_voltage: None,
            transit_time_current: 0.0,
            base_collector_capacitance: 0.0,
            base_collector_potential: 0.75,
            base_collector_grading: 0.33,
            internal_base_fraction: 1.0,
            reverse_transit_time: 0.0,
            depletion_coefficient: 0.5,
            substrate: None,
            substrate_saturation_current: None,
            substrate_emission_coefficient: None,
            substrate_capacitance: None,
            substrate_potential: None,
            substrate_grading: None,
        }
    }
}

/// The DC currents of the two junctions and the base charge they share.
#[derive(Debug, Clone, Copy)]
struct Junctions {
    saturation_current: f64,
    forward_beta: f64,
    reverse_beta: f64,
    forward_slope: f64,
    reverse_slope: f64,
    emitter_leakage: f64,
    emitter_leakage_slope: f64,
    collector_leakage: f64,
    collector_leakage_slope: f64,
    /// `1 / VAF` and `1 / VAR`, zero for infinite Early voltages.
    inverse_early: [f64; 2],
    /// `1 / IKF` and `1 / IKR`, zero for infinite knee currents.
    inverse_knee: [f64; 2],
}

/// The junctions' DC state at one bias: each value and its derivatives with respect to
/// the base-emitter and the base-collector voltages.
#[derive(Debug, Clone, Copy)]
struct State {
    /// The ideal base-emitter and base-collector currents, `Ibe` and `Ibc`, and their
    /// derivatives with respect to their own junction's voltage.
    ideal: [(f64, f64); 2],
    /// The base current through each junction, to the emitter and to the collector, and
    /// its derivative with respect to its junction's voltage.
    base: [(f64, f64); 2],
    /// The transport current from collector to emitter, and its derivatives.
    transport: (f64, [f64; 2]),
    /// The normalized base charge, `qb`, and its derivatives.
    charge: (f64, [f64; 2]),
}

impl Junctions {
    fn evaluate(&self, vbe: f64, vbc: f64, gmin: f64) -> State {
        let leakage = |voltage, current, slope| {
            if current > 0.0 {
                junction_current(voltage, current, slope)
            } else {
                (0.0, 0.0)
            }
        };
        let forward = junction_current(vbe, self.saturation_current, self.forward_slope);
        let reverse = junction_current(vbc, self.saturation_current, self.reverse_slope);
        let emitter_leakage = leakage(vbe, self.emitter_leakage, self.emitter_leakage_slope);
        let collector_leakage = leakage(vbc, self.collector_leakage, self.collector_leakage_slope);
        let base = [
            (
                forward.0 / self.forward_beta + emitter_leakage.0 + gmin * vbe,
                forward.1 / self.forward_beta + emitter_leakage.1 + gmin,
            ),
            (
                reverse.0 / self.reverse_beta + collector_leakage.0 + gmin * vbc,
                reverse.1 / self.reverse_beta + collector_leakage.1 + gmin,
            ),
        ];

        // The base charge: the Early effect through q1, high injection through q2.
        let [early_forward, early_reverse] = self.inverse_early;
        let [knee_forward, knee_reverse] = self.inverse_knee;
        let q1 = 1.0 / (1.0 - early_forward * vbc - early_reverse * vbe);
        let (qb, dqb) = if knee_forward == 0.0 && knee_reverse == 0.0 {
            (q1, [q1 * q1 * early_reverse, q1 * q1 * early_forward])
        } else {
            let q2 = knee_forward * forward.0 + knee_reverse * reverse.0;
            let root = (1.0 + 4.0 * q2).max(0.0).sqrt();
            // SPICE keeps the root at 1 where high injection would make it vanish.
            let root = if root == 0.0 { 1.0 } else { root };
            let qb = q1 * (1.0 + root) / 2.0;
            (
                qb,
                [
                    q1 * (qb * early_reverse + knee_forward * forward.1 / root),
                    q1 * (qb * early_forward + knee_reverse * reverse.1 / root),
                ],
            )
        };
        let net = forward.0 - reverse.0;
        let transport = (
            net / qb,
            [
                (forward.1 - net * dqb[0] / qb) / qb,
                (-reverse.1 - net * dqb[1] / qb) / qb,
            ],
        );
        State {
            ideal: [forward, reverse],
            base,
            transport,
            charge: (qb, dqb),
        }
    }
}

/// The base resistance, which falls from its zero-bias value towards its minimum.
#[derive(Debug, Clone, Copy)]
struct BaseResistance {
    maximum: f64,
    minimum: f64,
    /// The base current at which the resistance falls halfway, or `None` for a resistance
    /// that falls with the base charge.
    halfway_current: Option<f64>,
}

impl BaseResistance {
    /// The resistance and its derivatives with respect to the junction voltages, at the
    /// junctions' `state`.
    fn evaluate(&self, state: &State) -> (f64, [f64; 2]) {
        let excess = self.maximum - self.minimum;
        let Some(halfway) = self.halfway_current else {
            let (qb, dqb) = state.charge;
            let resistance = self.minimum + excess / qb;
            let slope = -excess / (qb * qb);
            return (resistance, [slope * dqb[0], slope * dqb[1]]);
        };
        let [(base_emitter, gbe), (base_collector, gbc)] = state.base;
        let ratio = (base_emitter + base_collector) / halfway;
        if ratio <= 1e-9 {
            // SPICE floors the ratio here, where the resistance no longer changes.
            let z = z_of(1e-9).0;
            return (self.minimum + 3.0 * excess * shape(z).0, [0.0, 0.0]);
        }
        let (z, dz) = z_of(ratio);
        let (value, dvalue) = shape(z);
        let slope = 3.0 * excess * dvalue * dz / halfway;
        (
            self.minimum + 3.0 * excess * value,
            [slope * gbe, slope * gbc],
        )
    }
}

/// SPICE's `z` for a base current `ratio` of the halfway current, and its derivative:
/// `z = (√(1 + 144 x / π²) − 1) / ((24 / π²) · √x)`.
fn z_of(ratio: f64) -> (f64, f64) {
    const A: f64 = 14.590_25;
    const B: f64 = 2.4317;
    let root = (1.0 + A * ratio).sqrt();
    let sqrt_ratio = ratio.sqrt();
    let z = (root - 1.0) / (B * sqrt_ratio);
    let dz = (A / (2.0 * root) * B * sqrt_ratio - (root - 1.0) * B / (2.0 * sqrt_ratio))
        / (B * B * ratio);
    (z, dz)
}

/// `(tan z − z) / (z · tan² z)`, and its derivative with respect to `z`.
///
/// Below `z = 0.25`, where the subtraction loses `3ε / z²` of the precision, a series in
/// `z²` takes over, accurate to within a few ulps there.
fn shape(z: f64) -> (f64, f64) {
    const SERIES: [f64; 7] = [
        1.0 / 3.0,
        -4.0 / 45.0,
        -4.0 / 315.0,
        -8.0 / 4725.0,
        -4.0 / 18711.0,
        -5528.0 / 212_837_625.0,
        -8.0 / 2_606_175.0,
    ];
    if z < 0.25 {
        // The value's series in `w = z²` by Horner's rule, and its derivative,
        // `z · Σ 2k · c_k · w^(k−1)`.
        let square = z * z;
        let value = SERIES
            .iter()
            .rev()
            .fold(0.0, |sum, coefficient| sum * square + coefficient);
        let orders = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let slope = SERIES
            .iter()
            .zip(orders)
            .skip(1)
            .rev()
            .fold(0.0, |sum, (coefficient, order)| {
                sum * square + order * coefficient
            });
        return (value, slope * z);
    }
    let tan = z.tan();
    let sec2 = 1.0 + tan * tan;
    let (top, top_slope) = (tan - z, sec2 - 1.0);
    let (bottom, bottom_slope) = (z * tan * tan, tan * tan + 2.0 * z * tan * sec2);
    (
        top / bottom,
        (top_slope * bottom - top * bottom_slope) / (bottom * bottom),
    )
}

/// What the transistor stores: the junctions' depletion charges and transit times.
#[derive(Debug, Clone, Copy)]
struct Charges {
    base_emitter: Option<Depletion>,
    /// The base-collector depletion on the internal base, and on the external one.
    base_collector: Option<Depletion>,
    external_base_collector: Option<Depletion>,
    forward_transit_time: f64,
    reverse_transit_time: f64,
    bias_coefficient: f64,
    /// `1 / (1.44 · VTF)`, zero for an infinite `VTF`.
    voltage_factor: f64,
    high_current: f64,
}

impl Charges {
    /// The base-emitter charge, its capacitance and its derivative with respect to the
    /// base-collector voltage, at the junctions' `state`.
    fn base_emitter(&self, vbe: f64, vbc: f64, state: &State) -> (f64, f64, f64) {
        let (mut current, mut conductance) = state.ideal[0];
        let mut cross = 0.0;
        let transit = self.forward_transit_time;
        if transit != 0.0 && vbe > 0.0 {
            let (qb, dqb) = state.charge;
            let (mut factor, mut growth, mut voltage_term) = (0.0, 0.0, 0.0);
            if self.bias_coefficient != 0.0 {
                factor = self.bias_coefficient * (vbc * self.voltage_factor).exp();
                growth = factor;
                if self.high_current != 0.0 {
                    let share = current / (current + self.high_current);
                    factor *= share * share;
                    growth = factor * (3.0 - 2.0 * share);
                }
                voltage_term = current * factor * self.voltage_factor;
            }
            let modulated = current * (1.0 + factor) / qb;
            conductance = (conductance * (1.0 + growth) - modulated * dqb[0]) / qb;
            cross = transit * (voltage_term - modulated * dqb[1]) / qb;
            current = modulated;
        }
        let (charge, capacitance) = depletion_charge(self.base_emitter.as_ref(), vbe);
        (
            transit * current + charge,
            transit * conductance + capacitance,
            cross,
        )
    }

    /// The internal base-collector charge and its capacitance.
    fn base_collector(&self, vbc: f64, state: &State) -> (f64, f64) {
        let (current, conductance) = state.ideal[1];
        let (charge, capacitance) = depletion_charge(self.base_collector.as_ref(), vbc);
        (
            self.reverse_transit_time * current + charge,
            self.reverse_transit_time * conductance + capacitance,
        )
    }
}

/// The junction from the substrate to the internal collector of a vertical transistor or
/// to the internal base of a lateral one.
#[derive(Debug, Clone, Copy)]
struct SubstrateJunction {
    construction: Construction,
    /// `±1`: orients the junction's voltage and current, from the substrate, as a vertical
    /// NPN's.
    sign: f64,
    /// The saturation current and exponential slope of the junction's diode, if it has one.
    diode: Option<(f64, f64)>,
    critical_voltage: f64,
    /// The depletion, whose capacitance grows linearly from zero bias.
    depletion: Option<Depletion>,
}

/// A transistor's substrate junction, between the nodes it joins.
#[derive(Debug)]
struct Substrate {
    /// The substrate, then the internal collector or base.
    stamp: Stamp<2>,
    junction: SubstrateJunction,
    /// The junction voltage of the last evaluation, from which the next step is limited.
    last: f64,
}

/// The critical voltage of a substrate junction without a diode, from which ngspice limits
/// its steps all the same.
const DIODELESS_CRITICAL_VOLTAGE: f64 = 50.0;

impl Substrate {
    /// Adds the junction's current and its charge.
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        let (stamp, junction) = (&self.stamp, &self.junction);
        let applied = junction.sign * (stamp.voltage(input, 0) - stamp.voltage(input, 1));
        let limited = limit_junction(
            applied,
            self.last,
            THERMAL_VOLTAGE,
            junction.critical_voltage,
        )
        .filter(|_| input.limiting);
        if limited.is_some() {
            output.mark_limited();
        }
        let voltage = limited.unwrap_or(applied);
        self.last = voltage;
        let beyond = applied - voltage;
        let (current, conductance) = junction.diode.map_or((0.0, 0.0), |(saturation, slope)| {
            junction_current(voltage, saturation, slope)
        });
        let (current, conductance) = (current + input.gmin * voltage, conductance + input.gmin);
        let pair = (0, 1);
        stamp.current(
            output,
            0,
            1,
            junction.sign * (current + conductance * beyond),
        );
        stamp.conductance(output, pair, pair, conductance);
        if let Some(depletion) = &junction.depletion {
            let (charge, capacitance) = depletion.evaluate(voltage);
            stamp.charge(
                output,
                0,
                1,
                junction.sign * (charge + capacitance * beyond),
            );
            stamp.capacitance(output, pair, pair, capacitance);
        }
    }
}

impl BipolarTransistorParameters {
    /// The junctions' DC model, once each of its parameters is checked.
    fn junctions(&self) -> Result<Junctions, ParameterError> {
        let positive = ParameterError::positive;
        let non_negative = ParameterError::non_negative;
        positive("saturation_current", self.saturation_current)?;
        positive("forward_beta", self.forward_beta)?;
        positive("reverse_beta", self.reverse_beta)?;
        positive(
            "forward_emission_coefficient",
            self.forward_emission_coefficient,
        )?;
        positive(
            "reverse_emission_coefficient",
            self.reverse_emission_coefficient,
        )?;
        non_negative(
            "base_emitter_leakage_current",
            self.base_emitter_leakage_current,
        )?;
        positive(
            "base_emitter_leakage_emission_coefficient",
            self.base_emitter_leakage_emission_coefficient,
        )?;
        non_negative(
            "base_collector_leakage_current",
            self.base_collector_leakage_current,
        )?;
        positive(
            "base_collector_leakage_emission_coefficient",
            self.base_collector_leakage_emission_coefficient,
        )?;
        Ok(Junctions {
            saturation_current: self.saturation_current,
            forward_beta: self.forward_beta,
            reverse_beta: self.reverse_beta,
            forward_slope: self.forward_emission_coefficient * THERMAL_VOLTAGE,
            reverse_slope: self.reverse_emission_coefficient * THERMAL_VOLTAGE,
            emitter_leakage: self.base_emitter_leakage_current,
            emitter_leakage_slope: self.base_emitter_leakage_emission_coefficient * THERMAL_VOLTAGE,
            collector_leakage: self.base_collector_leakage_current,
            collector_leakage_slope: self.base_collector_leakage_emission_coefficient
                * THERMAL_VOLTAGE,
            inverse_early: [
                inverse("forward_early_voltage", self.forward_early_voltage)?,
                inverse("reverse_early_voltage", self.reverse_early_voltage)?,
            ],
            inverse_knee: [
                inverse("forward_knee_current", self.forward_knee_current)?,
                inverse("reverse_knee_current", self.reverse_knee_current)?,
            ],
        })
    }

    /// The base resistance, if the transistor has one, once its parameters are checked.
    fn base_resistance(&self) -> Result<Option<BaseResistance>, ParameterError> {
        ParameterError::non_negative("base_resistance", self.base_resistance)?;
        let minimum = self.minimum_base_resistance.unwrap_or(self.base_resistance);
        ParameterError::non_negative("minimum_base_resistance", minimum)?;
        if let Some(current) = self.base_resistance_current {
            ParameterError::positive("base_resistance_current", current)?;
        }
        Ok((self.base_resistance > 0.0).then_some(BaseResistance {
            maximum: self.base_resistance,
            minimum,
            halfway_current: self.base_resistance_current,
        }))
    }

    /// What the transistor stores, if anything, once each of its parameters is checked.
    fn charges(&self) -> Result<Option<Charges>, ParameterError> {
        let non_negative = ParameterError::non_negative;
        non_negative("base_emitter_capacitance", self.base_emitter_capacitance)?;
        ParameterError::positive("base_emitter_potential", self.base_emitter_potential)?;
        below_one("base_emitter_grading", self.base_emitter_grading)?;
        non_negative("forward_transit_time", self.forward_transit_time)?;
        non_negative(
            "transit_time_bias_coefficient",
            self.transit_time_bias_coefficient,
        )?;
        non_negative("transit_time_current", self.transit_time_current)?;
        non_negative(
            "base_collector_capacitance",
            self.base_collector_capacitance,
        )?;
        ParameterError::positive("base_collector_potential", self.base_collector_potential)?;
        below_one("base_collector_grading", self.base_collector_grading)?;
        if !(0.0..=1.0).contains(&self.internal_base_fraction) {
            return Err(ParameterError(format!(
                "internal_base_fraction must be at least 0 and at most 1, not {:?}",
                self.internal_base_fraction
            )));
        }
        non_negative("reverse_transit_time", self.reverse_transit_time)?;
        below_one("depletion_coefficient", self.depletion_coefficient)?;
        let voltage_factor = inverse("transit_time_voltage", self.transit_time_voltage)? / 1.44;
        let stores = [
            self.base_emitter_capacitance,
            self.base_collector_capacitance,
            self.forward_transit_time,
            self.reverse_transit_time,
        ]
        .iter()
        .any(|value| *value > 0.0);
        let internal = self.base_collector_capacitance * self.internal_base_fraction;
        let collector_depletion = |capacitance| {
            Depletion::new(
                capacitance,
                self.base_collector_potential,
                self.base_collector_grading,
                self.depletion_coefficient,
            )
        };
        Ok(stores.then(|| Charges {
            base_emitter: Depletion::new(
                self.base_emitter_capacitance,
                self.base_emitter_potential,
                self.base_emitter_grading,
                self.depletion_coefficient,
            ),
            base_collector: collector_depletion(internal),
            external_base_collector: collector_depletion(
                self.base_collector_capacitance - internal,
            ),
            forward_transit_time: self.forward_transit_time,
            reverse_transit_time: self.reverse_transit_time,
            bias_coefficient: self.transit_time_bias_coefficient,
            voltage_factor,
            high_current: self.transit_time_current,
        }))
    }
}

impl BipolarTransistorParameters {
    /// The substrate junction, if the transistor has a substrate, once each of its
    /// parameters is checked.
    fn substrate_junction(&self) -> Result<Option<SubstrateJunction>, ParameterError> {
        let Some(construction) = self.substrate else {
            let given = [
                (
                    "substrate_saturation_current",
                    self.substrate_saturation_current,
                ),
                (
                    "substrate_emission_coefficient",
                    self.substrate_emission_coefficient,
                ),
                ("substrate_capacitance", self.substrate_capacitance),
                ("substrate_potential", self.substrate_potential),
                ("substrate_grading", self.substrate_grading),
            ]
            .into_iter()
            .find(|(_, value)| value.is_some());
            return match given {
                Some((name, _)) => Err(ParameterError(format!(
                    "{name} needs a substrate, vertical or lateral"
                ))),
                None => Ok(None),
            };
        };
        let emission = self.substrate_emission_coefficient.unwrap_or(1.0);
        ParameterError::positive("substrate_emission_coefficient", emission)?;
        let capacitance = self.substrate_capacitance.unwrap_or(0.0);
        ParameterError::non_negative("substrate_capacitance", capacitance)?;
        let potential = self.substrate_potential.unwrap_or(0.75);
        ParameterError::positive("substrate_potential", potential)?;
        let grading = below_one("substrate_grading", self.substrate_grading.unwrap_or(0.0))?;
        let diode = self
            .substrate_saturation_current
            .map(|current| {
                ParameterError::positive("substrate_saturation_current", current)
                    .map(|current| (current, emission * THERMAL_VOLTAGE))
            })
            .transpose()?;
        let orientation = match construction {
            Construction::Vertical => 1.0,
            Construction::Lateral => -1.0,
        };
        Ok(Some(SubstrateJunction {
            construction,
            sign: self.polarity.sign() * orientation,
            diode,
            critical_voltage: diode.map_or(DIODELESS_CRITICAL_VOLTAGE, |(current, _)| {
                critical_voltage(current, THERMAL_VOLTAGE)
            }),
            depletion: Depletion::new(capacitance, potential, grading, 0.0),
        }))
    }
}

/// The reciprocal of an optional positive parameter, zero when it is left out, infinite.
fn inverse(name: &str, value: Option<f64>) -> Result<f64, ParameterError> {
    value.map_or(Ok(0.0), |value| {
        ParameterError::positive(name, value).map(f64::recip)
    })
}

impl NativeDevice for BipolarTransistor {
    const NAME: &'static str = "bipolar_transistor";
    type Parameters = BipolarTransistorParameters;

    fn interface(parameters: &BipolarTransistorParameters) -> Interface {
        let mut terminals = vec!["collector", "base", "emitter"];
        let mut branches = vec![
            Branch::new(1, 2, BranchKind::Conductive),
            Branch::new(1, 0, BranchKind::Conductive),
            Branch::new(0, 2, BranchKind::Conductive),
        ];
        if let Some(construction) = parameters.substrate {
            let joined = match construction {
                Construction::Vertical => COLLECTOR,
                Construction::Lateral => BASE,
            };
            terminals.push("substrate");
            branches.push(Branch::new(3, joined, BranchKind::Conductive));
        }
        Interface::new(&terminals, &branches)
    }

    fn new(
        parameters: BipolarTransistorParameters,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let junctions = parameters.junctions()?;
        let base = parameters.base_resistance()?;
        let charges = parameters.charges()?;
        let substrate_junction = parameters.substrate_junction()?;
        let non_negative = ParameterError::non_negative;
        let collector_resistance =
            non_negative("collector_resistance", parameters.collector_resistance)?;
        let emitter_resistance = non_negative("emitter_resistance", parameters.emitter_resistance)?;

        let [collector, base_terminal, emitter] = [0, 1, 2].map(|index| binding.terminal(index));
        let mut inner = |terminal, resistance: f64, label| {
            if resistance > 0.0 {
                Some(binding.internal(label, Quantity::Voltage))
            } else {
                terminal
            }
        };
        let inner_collector = inner(collector, collector_resistance, "collector");
        let inner_base = inner(base_terminal, parameters.base_resistance, "base");
        let inner_emitter = inner(emitter, emitter_resistance, "emitter");
        let substrate = substrate_junction.map(|junction| {
            let joined = match junction.construction {
                Construction::Vertical => inner_collector,
                Construction::Lateral => inner_base,
            };
            let nodes = [binding.terminal(3), joined];
            Substrate {
                stamp: Stamp::new(binding, nodes),
                junction,
                last: 0.0,
            }
        });
        let stamp = Stamp::new(
            binding,
            [
                collector,
                base_terminal,
                emitter,
                inner_collector,
                inner_base,
                inner_emitter,
            ],
        );
        let conductance = |resistance: f64| {
            if resistance > 0.0 {
                1.0 / resistance
            } else {
                0.0
            }
        };
        Ok(Self {
            stamp,
            sign: parameters.polarity.sign(),
            collector_conductance: conductance(collector_resistance),
            emitter_conductance: conductance(emitter_resistance),
            base,
            critical_voltage: critical_voltage(junctions.saturation_current, THERMAL_VOLTAGE),
            junctions,
            charges,
            substrate,
            last: [0.0; 2],
        })
    }
}

/// The junction voltages one evaluation uses: limited, and how far the applied ones lie
/// beyond them, along which every value is extended linearly.
#[derive(Debug, Clone, Copy)]
struct Bias {
    /// The base-emitter and base-collector voltages, limited.
    voltages: [f64; 2],
    beyond: [f64; 2],
}

impl Bias {
    /// How far a value with derivatives `slopes` moves from the limited voltages to the
    /// applied ones.
    fn extend(&self, slopes: [f64; 2]) -> f64 {
        slopes[0] * self.beyond[0] + slopes[1] * self.beyond[1]
    }
}

const BASE_EMITTER: (usize, usize) = (INNER_BASE, INNER_EMITTER);
const BASE_COLLECTOR: (usize, usize) = (INNER_BASE, INNER_COLLECTOR);

impl BipolarTransistor {
    /// Adds the collector and emitter resistances.
    fn add_series(&self, input: &Input<'_>, output: &mut Output<'_>) {
        let stamp = &self.stamp;
        for (outer, inner, conductance) in [
            (COLLECTOR, INNER_COLLECTOR, self.collector_conductance),
            (EMITTER, INNER_EMITTER, self.emitter_conductance),
        ] {
            if conductance > 0.0 {
                let voltage = stamp.voltage(input, outer) - stamp.voltage(input, inner);
                stamp.current(output, outer, inner, conductance * voltage);
                stamp.conductance(output, (outer, inner), (outer, inner), conductance);
            }
        }
    }

    /// The junction voltages at `input`, limited from the last evaluation's.
    fn bias(&mut self, input: &Input<'_>, output: &mut Output<'_>) -> Bias {
        let stamp = &self.stamp;
        let base = stamp.voltage(input, INNER_BASE);
        let applied = [
            self.sign * (base - stamp.voltage(input, INNER_EMITTER)),
            self.sign * (base - stamp.voltage(input, INNER_COLLECTOR)),
        ];
        let limits = [0, 1].map(|index| {
            limit_junction(
                applied[index],
                self.last[index],
                THERMAL_VOLTAGE,
                self.critical_voltage,
            )
            .filter(|_| input.limiting)
        });
        if limits.iter().any(Option::is_some) {
            output.mark_limited();
        }
        let voltages = [0, 1].map(|index| limits[index].unwrap_or(applied[index]));
        self.last = voltages;
        Bias {
            voltages,
            beyond: [applied[0] - voltages[0], applied[1] - voltages[1]],
        }
    }

    /// Adds the transport current and the base currents through each junction.
    fn add_currents(&self, bias: &Bias, state: &State, output: &mut Output<'_>) {
        let (stamp, sign) = (&self.stamp, self.sign);
        let (transport, slopes) = state.transport;
        let collector_emitter = (INNER_COLLECTOR, INNER_EMITTER);
        let current = sign * (transport + bias.extend(slopes));
        stamp.current(output, INNER_COLLECTOR, INNER_EMITTER, current);
        stamp.conductance(output, collector_emitter, BASE_EMITTER, slopes[0]);
        stamp.conductance(output, collector_emitter, BASE_COLLECTOR, slopes[1]);
        for (index, pair) in [BASE_EMITTER, BASE_COLLECTOR].into_iter().enumerate() {
            let (current, conductance) = state.base[index];
            let current = sign * (current + conductance * bias.beyond[index]);
            stamp.current(output, pair.0, pair.1, current);
            stamp.conductance(output, pair, pair, conductance);
        }
    }

    /// Adds the current through the base resistance, `drop / r`, which varies with the
    /// drop and with the junction voltages the resistance depends on.
    fn add_base_resistance(
        &self,
        resistance: &BaseResistance,
        bias: &Bias,
        state: &State,
        input: &Input<'_>,
        output: &mut Output<'_>,
    ) {
        let stamp = &self.stamp;
        let (value, slopes) = resistance.evaluate(state);
        let drop = stamp.voltage(input, BASE) - stamp.voltage(input, INNER_BASE);
        let slopes = slopes.map(|slope| -drop * slope / (value * value));
        let pair = (BASE, INNER_BASE);
        stamp.current(output, BASE, INNER_BASE, drop / value + bias.extend(slopes));
        stamp.conductance(output, pair, pair, 1.0 / value);
        stamp.conductance(output, pair, BASE_EMITTER, self.sign * slopes[0]);
        stamp.conductance(output, pair, BASE_COLLECTOR, self.sign * slopes[1]);
    }

    /// Adds the charges of the junctions, and the external base-collector depletion,
    /// whose voltage SPICE never limits.
    fn add_charges(
        &self,
        charges: &Charges,
        bias: &Bias,
        state: &State,
        input: &Input<'_>,
        output: &mut Output<'_>,
    ) {
        let (stamp, sign) = (&self.stamp, self.sign);
        let [vbe, vbc] = bias.voltages;
        let (charge, capacitance, cross) = charges.base_emitter(vbe, vbc, state);
        let charge = sign * (charge + bias.extend([capacitance, cross]));
        stamp.charge(output, INNER_BASE, INNER_EMITTER, charge);
        stamp.capacitance(output, BASE_EMITTER, BASE_EMITTER, capacitance);
        stamp.capacitance(output, BASE_EMITTER, BASE_COLLECTOR, cross);
        let (charge, capacitance) = charges.base_collector(vbc, state);
        let charge = sign * (charge + capacitance * bias.beyond[1]);
        stamp.charge(output, INNER_BASE, INNER_COLLECTOR, charge);
        stamp.capacitance(output, BASE_COLLECTOR, BASE_COLLECTOR, capacitance);
        let external = sign * (stamp.voltage(input, BASE) - stamp.voltage(input, INNER_COLLECTOR));
        let (charge, capacitance) =
            depletion_charge(charges.external_base_collector.as_ref(), external);
        let pair = (BASE, INNER_COLLECTOR);
        stamp.charge(output, BASE, INNER_COLLECTOR, sign * charge);
        stamp.capacitance(output, pair, pair, capacitance);
    }
}

impl Device for BipolarTransistor {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_>) {
        self.add_series(input, output);
        let bias = self.bias(input, output);
        let [vbe, vbc] = bias.voltages;
        let state = self.junctions.evaluate(vbe, vbc, input.gmin);
        self.add_currents(&bias, &state, output);
        if let Some(resistance) = &self.base {
            self.add_base_resistance(resistance, &bias, &state, input, output);
        }
        if let Some(charges) = &self.charges {
            self.add_charges(charges, &bias, &state, input, output);
        }
        if let Some(substrate) = &mut self.substrate {
            substrate.evaluate(input, output);
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{Bench, assert_jacobians_match};

    /// A transistor with every effect switched on.
    const FULL: &str = r#"{"saturation_current": 1e-15, "forward_beta": 150,
        "forward_early_voltage": 60, "reverse_early_voltage": 20,
        "forward_knee_current": 0.05, "reverse_knee_current": 0.01,
        "base_emitter_leakage_current": 1e-14, "base_collector_leakage_current": 1e-13,
        "base_emitter_capacitance": 2e-12, "base_collector_capacitance": 1e-12,
        "base_collector_grading": 0.4, "internal_base_fraction": 0.7,
        "forward_transit_time": 4e-10, "transit_time_bias_coefficient": 2,
        "transit_time_voltage": 3, "transit_time_current": 0.02,
        "reverse_transit_time": 1e-8}"#;

    #[test]
    fn its_collector_current_is_beta_times_its_base_current() {
        let mut bench = Bench::new::<BipolarTransistor>(r#"{"saturation_current": 1e-15}"#);
        let evaluation = bench.settle(&[5.0, 0.65, 0.0]);
        let (collector, base) = (evaluation.residual[0], evaluation.residual[1]);
        let expected = 1e-15 * ((0.65 / THERMAL_VOLTAGE).exp() - 1.0);
        // The base-collector junction's reverse current adds a little to both.
        assert!(
            (collector / expected - 1.0).abs() < 1e-6,
            "{collector} vs {expected}"
        );
        assert!(
            (collector / base - 100.0).abs() < 1e-3,
            "{}",
            collector / base
        );
    }

    #[test]
    fn a_pnp_mirrors_an_npn() {
        let mut npn = Bench::new::<BipolarTransistor>(FULL);
        let mut pnp =
            Bench::new::<BipolarTransistor>(&FULL.replace('{', r#"{"polarity": "pnp", "#));
        let n = npn.settle(&[3.0, 0.7, 0.0]);
        let p = pnp.settle(&[-3.0, -0.7, 0.0]);
        for (n, p) in n.residual.iter().zip(&p.residual) {
            assert!((n + p).abs() <= 1e-15 * n.abs().max(1.0));
        }
        for (n, p) in n.charge.iter().zip(&p.charge) {
            assert!((n + p).abs() <= 1e-15 * n.abs().max(1e-12));
        }
    }

    #[test]
    fn series_resistances_add_internal_nodes() {
        let bench = Bench::new::<BipolarTransistor>(
            r#"{"base_resistance": 100, "collector_resistance": 5, "emitter_resistance": 1}"#,
        );
        assert_eq!(
            bench.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "device:collector",
                "device:base",
                "device:emitter"
            ]
        );
    }

    #[test]
    fn the_base_resistance_shape_is_continuous_where_its_series_hands_over() {
        let z = 0.25 - 1e-12;
        let (below, below_slope) = shape(z);
        let tan = z.tan();
        let direct = (tan - z) / (z * tan * tan);
        assert!((below / direct - 1.0).abs() < 1e-14, "{below} vs {direct}");
        let (above, above_slope) = shape(0.25 + 1e-12);
        assert!((above / below - 1.0).abs() < 1e-11);
        assert!((above_slope / below_slope - 1.0).abs() < 1e-9);
        assert!((shape(1e-9).0 - 1.0 / 3.0).abs() < 1e-17);
    }

    #[test]
    fn a_substrate_is_a_fourth_terminal_joined_to_the_collector_or_the_base() {
        let parameters = |construction| {
            serde_json::from_str::<BipolarTransistorParameters>(&format!(
                r#"{{"substrate": "{construction}"}}"#
            ))
            .unwrap()
        };
        let vertical = BipolarTransistor::interface(&parameters("vertical"));
        assert_eq!(
            vertical.terminals,
            ["collector", "base", "emitter", "substrate"]
        );
        assert_eq!(
            vertical.branches[3],
            Branch::new(3, COLLECTOR, BranchKind::Conductive)
        );
        let lateral = BipolarTransistor::interface(&parameters("lateral"));
        assert_eq!(
            lateral.branches[3],
            Branch::new(3, BASE, BranchKind::Conductive)
        );
    }

    #[test]
    fn a_substrate_junction_without_a_diode_conducts_the_minimum_conductance() {
        let mut bench = Bench::new::<BipolarTransistor>(r#"{"substrate": "vertical"}"#);
        let (gmin, x) = (1e-12, [2.0, 0.5, 0.0, -3.0]);
        let evaluation = bench.evaluate_with(&Input::new(&x, 0.0, gmin));
        assert!((evaluation.residual[3] - gmin * (x[3] - x[0])).abs() < 1e-24);
        assert!((evaluation.conductance[3][3] - gmin).abs() < 1e-24);
    }

    #[test]
    fn a_substrate_junction_conducts_from_its_p_side() {
        // A lateral PNP's substrate is its junction's p side, a vertical one's its n side.
        for (construction, joined, forward) in
            [("lateral", BASE, 1.0), ("vertical", COLLECTOR, -1.0)]
        {
            let mut bench = Bench::new::<BipolarTransistor>(&format!(
                r#"{{"polarity": "pnp", "substrate": "{construction}",
                    "substrate_saturation_current": 1e-15}}"#
            ));
            let mut x = [-2.0, -0.5, 0.0, 0.0];
            x[3] = x[joined] + forward * 0.6;
            let current = bench.settle(&x).residual[3];
            let expected = forward * 1e-15 * ((0.6 / THERMAL_VOLTAGE).exp() - 1.0);
            assert!(
                (current / expected - 1.0).abs() < 1e-12,
                "{current} vs {expected}"
            );
        }
    }

    #[test]
    fn substrate_parameters_need_a_substrate() {
        let error =
            Bench::try_new::<BipolarTransistor>(r#"{"substrate_capacitance": 1e-12}"#).unwrap_err();
        assert_eq!(
            error.to_string(),
            "substrate_capacitance needs a substrate, vertical or lateral"
        );
    }

    #[test]
    fn rejects_out_of_range_parameters() {
        let error =
            Bench::try_new::<BipolarTransistor>(r#"{"forward_early_voltage": 0}"#).unwrap_err();
        assert_eq!(
            error.to_string(),
            "forward_early_voltage must be positive, not 0.0"
        );
    }

    proptest! {
        /// The conductances and capacitances are the derivatives of the currents and
        /// charges, forward, reverse and saturated, with the base resistance modulated by
        /// the base charge or by the base current. Neither junction is forward-biased
        /// beyond 0.85 V, where its current would be unphysical.
        #[test]
        fn jacobians_match_finite_differences(
            base in -1.0..0.85f64,
            collector_above_base in -0.85..5.0f64,
            pnp in proptest::bool::ANY,
            halfway in proptest::bool::ANY,
        ) {
            let collector = base + collector_above_base;
            let mut parameters = FULL.to_owned();
            if pnp {
                parameters = parameters.replace('{', r#"{"polarity": "pnp", "#);
            }
            let sign = if pnp { -1.0 } else { 1.0 };
            assert_jacobians_match::<BipolarTransistor>(
                &parameters,
                &[sign * collector, sign * base, 0.0],
            );
            let resistive = parameters.replace(
                '{',
                if halfway {
                    r#"{"base_resistance": 200, "minimum_base_resistance": 20,
                        "base_resistance_current": 1e-4, "collector_resistance": 10,
                        "emitter_resistance": 2, "#
                } else {
                    r#"{"base_resistance": 200, "minimum_base_resistance": 20,
                        "collector_resistance": 10, "emitter_resistance": 2, "#
                },
            );
            let substrate = resistive.replace(
                '{',
                r#"{"substrate": "lateral", "substrate_saturation_current": 1e-16,
                    "substrate_emission_coefficient": 1.2, "substrate_capacitance": 3e-12,
                    "substrate_potential": 0.6, "substrate_grading": 0.4, "#,
            );
            assert_jacobians_match::<BipolarTransistor>(
                &substrate,
                &[
                    sign * collector,
                    sign * base,
                    0.0,
                    sign * (base - 0.4),
                    sign * 0.98 * collector,
                    sign * 0.97 * base,
                    sign * 0.01,
                ],
            );
            assert_jacobians_match::<BipolarTransistor>(
                &resistive,
                &[
                    sign * collector,
                    sign * base,
                    0.0,
                    sign * 0.98 * collector,
                    sign * 0.97 * base,
                    sign * 0.01,
                ],
            );
        }
    }
}
