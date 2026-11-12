use serde::Deserialize;

use crate::junction::{
    Depletion, Temperatures, below_one, critical_voltage, depletion_charge, junction_current,
    limit_junction,
};
use crate::stamp::{self, Stamp, series_conductance};
use crate::{
    Binding, Branch, BranchKind, Device, Input, Interface, NativeDevice, NativeModel, Output,
    ParameterError, REFERENCE_TEMPERATURE, Real,
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
///
/// Away from its nominal temperature, its saturation currents scale with the energy gap,
/// its betas by their temperature exponent, and its junctions' potentials and
/// capacitances with silicon's band gap, as ngspice scales them.
#[derive(Debug)]
pub struct BipolarTransistor<R: Real = f64> {
    stamp: Stamp<6>,
    sign: f64,
    collector_conductance: R,
    emitter_conductance: R,
    base: Option<BaseResistance<R>>,
    junctions: Junctions<R>,
    charges: Option<Charges<R>>,
    substrate: Option<Substrate<R>>,
    /// The thermal voltage at the transistor's temperature.
    thermal_voltage: f64,
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
#[serde(default, deny_unknown_fields, bound = "")]
pub struct BipolarTransistorParameters<R: Real = f64> {
    /// The doping order; default `npn`.
    pub polarity: Polarity,
    /// The transport saturation current, `IS`, in amperes; default 1e-16.
    pub saturation_current: R,
    /// The ideal maximum forward current gain, `BF`; default 100.
    pub forward_beta: R,
    /// The ideal maximum reverse current gain, `BR`; default 1.
    pub reverse_beta: R,
    /// The forward emission coefficient, `NF`; default 1.
    pub forward_emission_coefficient: R,
    /// The reverse emission coefficient, `NR`; default 1.
    pub reverse_emission_coefficient: R,
    /// The forward Early voltage, `VAF`, in volts; infinite by default.
    pub forward_early_voltage: Option<R>,
    /// The reverse Early voltage, `VAR`, in volts; infinite by default.
    pub reverse_early_voltage: Option<R>,
    /// The forward knee current, `IKF`, where high injection sets in, in amperes; infinite
    /// by default.
    pub forward_knee_current: Option<R>,
    /// The reverse knee current, `IKR`, in amperes; infinite by default.
    pub reverse_knee_current: Option<R>,
    /// The base-emitter leakage saturation current, `ISE`, in amperes; default 0.
    pub base_emitter_leakage_current: R,
    /// The base-emitter leakage emission coefficient, `NE`; default 1.5.
    pub base_emitter_leakage_emission_coefficient: R,
    /// The base-collector leakage saturation current, `ISC`, in amperes; default 0.
    pub base_collector_leakage_current: R,
    /// The base-collector leakage emission coefficient, `NC`; default 2.
    pub base_collector_leakage_emission_coefficient: R,
    /// The zero-bias base resistance, `RB`, in ohms; default 0.
    pub base_resistance: R,
    /// The base resistance at high current, `RBM`, in ohms; the base resistance by default.
    pub minimum_base_resistance: Option<R>,
    /// The base current, `IRB`, at which the base resistance falls halfway to its minimum,
    /// in amperes; without it, the base resistance falls with the base charge.
    pub base_resistance_current: Option<R>,
    /// The emitter resistance, `RE`, in ohms; default 0.
    pub emitter_resistance: R,
    /// The collector resistance, `RC`, in ohms; default 0.
    pub collector_resistance: R,
    /// The zero-bias base-emitter depletion capacitance, `CJE`, in farads; default 0.
    pub base_emitter_capacitance: R,
    /// The base-emitter built-in potential, `VJE`, in volts; default 0.75.
    pub base_emitter_potential: R,
    /// The base-emitter grading coefficient, `MJE`, below 1; default 0.33.
    pub base_emitter_grading: R,
    /// The ideal forward transit time, `TF`, in seconds; default 0.
    pub forward_transit_time: R,
    /// How much the forward transit time grows with bias, `XTF`; default 0.
    pub transit_time_bias_coefficient: R,
    /// The base-collector voltage, `VTF`, that sets how the forward transit time grows with
    /// it, in volts; infinite by default.
    pub transit_time_voltage: Option<R>,
    /// The current, `ITF`, above which the forward transit time grows, in amperes;
    /// default 0.
    pub transit_time_current: R,
    /// The zero-bias base-collector depletion capacitance, `CJC`, in farads; default 0.
    pub base_collector_capacitance: R,
    /// The base-collector built-in potential, `VJC`, in volts; default 0.75.
    pub base_collector_potential: R,
    /// The base-collector grading coefficient, `MJC`, below 1; default 0.33.
    pub base_collector_grading: R,
    /// The fraction of the base-collector capacitance on the internal base, `XCJC`, the
    /// rest on the external base; default 1.
    pub internal_base_fraction: R,
    /// The ideal reverse transit time, `TR`, in seconds; default 0.
    pub reverse_transit_time: R,
    /// The fraction of each junction's potential above which its depletion capacitance is
    /// extended linearly, `FC`, below 1; default 0.5.
    pub depletion_coefficient: R,
    /// How the transistor is built, if it has a substrate, as ngspice's `SUBS` of 1 and −1
    /// builds it; without it, the transistor has none. The parameters below, of the
    /// junction to the substrate, need one.
    pub substrate: Option<Construction>,
    /// The substrate junction's saturation current, `ISS`, in amperes; without it, the
    /// junction carries the minimum conductance only.
    pub substrate_saturation_current: Option<R>,
    /// The substrate junction's emission coefficient, `NS`; default 1.
    pub substrate_emission_coefficient: Option<R>,
    /// The substrate junction's zero-bias depletion capacitance, `CJS`, in farads;
    /// default 0.
    pub substrate_capacitance: Option<R>,
    /// The substrate junction's built-in potential, `VJS`, in volts; default 0.75.
    pub substrate_potential: Option<R>,
    /// The substrate junction's grading coefficient, `MJS`, below 1; default 0.
    pub substrate_grading: Option<R>,
    /// The energy gap that scales the saturation currents with temperature, `EG`, in
    /// volts; default 1.11, silicon's.
    pub energy_gap: R,
    /// The exponent of the temperature ratio in the saturation currents' scaling, `XTI`;
    /// default 3.
    pub saturation_current_temperature_exponent: R,
    /// The exponent of the temperature ratio that scales the betas, `XTB`; default 0.
    pub beta_temperature_exponent: R,
    /// The temperature the parameters were measured at, in kelvins; default 27 °C.
    pub nominal_temperature: R,
}

impl<R: Real> Default for BipolarTransistorParameters<R> {
    fn default() -> Self {
        Self {
            polarity: Polarity::Npn,
            saturation_current: R::constant(1e-16),
            forward_beta: R::constant(100.0),
            reverse_beta: R::constant(1.0),
            forward_emission_coefficient: R::constant(1.0),
            reverse_emission_coefficient: R::constant(1.0),
            forward_early_voltage: None,
            reverse_early_voltage: None,
            forward_knee_current: None,
            reverse_knee_current: None,
            base_emitter_leakage_current: R::constant(0.0),
            base_emitter_leakage_emission_coefficient: R::constant(1.5),
            base_collector_leakage_current: R::constant(0.0),
            base_collector_leakage_emission_coefficient: R::constant(2.0),
            base_resistance: R::constant(0.0),
            minimum_base_resistance: None,
            base_resistance_current: None,
            emitter_resistance: R::constant(0.0),
            collector_resistance: R::constant(0.0),
            base_emitter_capacitance: R::constant(0.0),
            base_emitter_potential: R::constant(0.75),
            base_emitter_grading: R::constant(0.33),
            forward_transit_time: R::constant(0.0),
            transit_time_bias_coefficient: R::constant(0.0),
            transit_time_voltage: None,
            transit_time_current: R::constant(0.0),
            base_collector_capacitance: R::constant(0.0),
            base_collector_potential: R::constant(0.75),
            base_collector_grading: R::constant(0.33),
            internal_base_fraction: R::constant(1.0),
            reverse_transit_time: R::constant(0.0),
            depletion_coefficient: R::constant(0.5),
            substrate: None,
            substrate_saturation_current: None,
            substrate_emission_coefficient: None,
            substrate_capacitance: None,
            substrate_potential: None,
            substrate_grading: None,
            energy_gap: R::constant(1.11),
            saturation_current_temperature_exponent: R::constant(3.0),
            beta_temperature_exponent: R::constant(0.0),
            nominal_temperature: R::constant(REFERENCE_TEMPERATURE),
        }
    }
}

/// The DC currents of the two junctions and the base charge they share.
#[derive(Debug, Clone, Copy)]
struct Junctions<R: Real> {
    saturation_current: R,
    forward_beta: R,
    reverse_beta: R,
    forward_slope: R,
    reverse_slope: R,
    emitter_leakage: R,
    emitter_leakage_slope: R,
    collector_leakage: R,
    collector_leakage_slope: R,
    /// `1 / VAF` and `1 / VAR`, zero for infinite Early voltages.
    inverse_early: [R; 2],
    /// `1 / IKF` and `1 / IKR`, zero for infinite knee currents.
    inverse_knee: [R; 2],
}

/// The junctions' DC state at one bias: each value and its derivatives with respect to
/// the base-emitter and the base-collector voltages.
#[derive(Debug, Clone, Copy)]
struct State<R: Real> {
    /// The ideal base-emitter and base-collector currents, `Ibe` and `Ibc`, and their
    /// derivatives with respect to their own junction's voltage.
    ideal: [(R, R); 2],
    /// The base current through each junction, to the emitter and to the collector, and
    /// its derivative with respect to its junction's voltage.
    base: [(R, R); 2],
    /// The transport current from collector to emitter, and its derivatives.
    transport: (R, [R; 2]),
    /// The normalized base charge, `qb`, and its derivatives.
    charge: (R, [R; 2]),
}

impl<R: Real> Junctions<R> {
    fn evaluate(&self, vbe: f64, vbc: f64, gmin: f64) -> State<R> {
        let zero = R::constant(0.0);
        let leakage = |voltage, current: R, slope| {
            if current.value() > 0.0 {
                junction_current(voltage, current, slope)
            } else {
                (zero, zero)
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
        let q1 = (-(early_forward * vbc - 1.0) - early_reverse * vbe).recip();
        let (qb, dqb) = if knee_forward.value() == 0.0 && knee_reverse.value() == 0.0 {
            (q1, [q1 * q1 * early_reverse, q1 * q1 * early_forward])
        } else {
            let q2 = knee_forward * forward.0 + knee_reverse * reverse.0;
            let root = (q2 * 4.0 + 1.0).max(zero).sqrt();
            // SPICE keeps the root at 1 where high injection would make it vanish.
            let root = if root.value() == 0.0 {
                R::constant(1.0)
            } else {
                root
            };
            let qb = q1 * (root + 1.0) / 2.0;
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
struct BaseResistance<R: Real> {
    maximum: R,
    minimum: R,
    /// The base current at which the resistance falls halfway, or `None` for a resistance
    /// that falls with the base charge.
    halfway_current: Option<R>,
}

impl<R: Real> BaseResistance<R> {
    /// The resistance and its derivatives with respect to the junction voltages, at the
    /// junctions' `state`.
    fn evaluate(&self, state: &State<R>) -> (R, [R; 2]) {
        let excess = self.maximum - self.minimum;
        let Some(halfway) = self.halfway_current else {
            let (qb, dqb) = state.charge;
            let resistance = self.minimum + excess / qb;
            let slope = -excess / (qb * qb);
            return (resistance, [slope * dqb[0], slope * dqb[1]]);
        };
        let [(base_emitter, gbe), (base_collector, gbc)] = state.base;
        let ratio = (base_emitter + base_collector) / halfway;
        if ratio.value() <= 1e-9 {
            // SPICE floors the ratio here, where the resistance no longer changes.
            let z = z_of(R::constant(1e-9)).0;
            let zero = R::constant(0.0);
            return (self.minimum + excess * 3.0 * shape(z).0, [zero, zero]);
        }
        let (z, dz) = z_of(ratio);
        let (value, dvalue) = shape(z);
        let slope = excess * 3.0 * dvalue * dz / halfway;
        (
            self.minimum + excess * 3.0 * value,
            [slope * gbe, slope * gbc],
        )
    }
}

/// SPICE's `z` for a base current `ratio` of the halfway current, and its derivative:
/// `z = (√(1 + 144 x / π²) − 1) / ((24 / π²) · √x)`.
fn z_of<R: Real>(ratio: R) -> (R, R) {
    const A: f64 = 14.590_25;
    const B: f64 = 2.4317;
    let root = (ratio * A + 1.0).sqrt();
    let sqrt_ratio = ratio.sqrt();
    let z = (root - 1.0) / (sqrt_ratio * B);
    let dz = (R::constant(A) / (root * 2.0) * B * sqrt_ratio
        - (root - 1.0) * B / (sqrt_ratio * 2.0))
        / (ratio * (B * B));
    (z, dz)
}

/// `(tan z − z) / (z · tan² z)`, and its derivative with respect to `z`.
///
/// Below `z = 0.25`, where the subtraction loses `3ε / z²` of the precision, a series in
/// `z²` takes over, accurate to within a few ulps there.
fn shape<R: Real>(z: R) -> (R, R) {
    const SERIES: [f64; 7] = [
        1.0 / 3.0,
        -4.0 / 45.0,
        -4.0 / 315.0,
        -8.0 / 4725.0,
        -4.0 / 18711.0,
        -5528.0 / 212_837_625.0,
        -8.0 / 2_606_175.0,
    ];
    if z.value() < 0.25 {
        // The value's series in `w = z²` by Horner's rule, and its derivative,
        // `z · Σ 2k · c_k · w^(k−1)`.
        let square = z * z;
        let value = SERIES
            .iter()
            .rev()
            .fold(R::constant(0.0), |sum, coefficient| {
                sum * square + *coefficient
            });
        let orders = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let slope = SERIES
            .iter()
            .zip(orders)
            .skip(1)
            .rev()
            .fold(R::constant(0.0), |sum, (coefficient, order)| {
                sum * square + order * coefficient
            });
        return (value, slope * z);
    }
    let tan = z.tan();
    let sec2 = tan * tan + 1.0;
    let (top, top_slope) = (tan - z, sec2 - 1.0);
    let (bottom, bottom_slope) = (z * tan * tan, tan * tan + z * 2.0 * tan * sec2);
    (
        top / bottom,
        (top_slope * bottom - top * bottom_slope) / (bottom * bottom),
    )
}

/// What the transistor stores: the junctions' depletion charges and transit times.
#[derive(Debug, Clone, Copy)]
struct Charges<R: Real> {
    base_emitter: Option<Depletion<R>>,
    /// The base-collector depletion on the internal base, and on the external one.
    base_collector: Option<Depletion<R>>,
    external_base_collector: Option<Depletion<R>>,
    forward_transit_time: R,
    reverse_transit_time: R,
    bias_coefficient: R,
    /// `1 / (1.44 · VTF)`, zero for an infinite `VTF`.
    voltage_factor: R,
    high_current: R,
}

impl<R: Real> Charges<R> {
    /// The base-emitter charge, its capacitance and its derivative with respect to the
    /// base-collector voltage, at the junctions' `state`.
    fn base_emitter(&self, vbe: f64, vbc: f64, state: &State<R>) -> (R, R, R) {
        let zero = R::constant(0.0);
        let (mut current, mut conductance) = state.ideal[0];
        let mut cross = zero;
        let transit = self.forward_transit_time;
        if transit.value() != 0.0 && vbe > 0.0 {
            let (qb, dqb) = state.charge;
            let (mut factor, mut growth, mut voltage_term) = (zero, zero, zero);
            if self.bias_coefficient.value() != 0.0 {
                factor = self.bias_coefficient * (self.voltage_factor * vbc).exp();
                growth = factor;
                if self.high_current.value() != 0.0 {
                    let share = current / (current + self.high_current);
                    factor = factor * (share * share);
                    growth = factor * -(share * 2.0 - 3.0);
                }
                voltage_term = current * factor * self.voltage_factor;
            }
            let modulated = current * (factor + 1.0) / qb;
            conductance = (conductance * (growth + 1.0) - modulated * dqb[0]) / qb;
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
    fn base_collector(&self, vbc: f64, state: &State<R>) -> (R, R) {
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
struct SubstrateJunction<R: Real> {
    construction: Construction,
    /// `±1`: orients the junction's voltage and current, from the substrate, as a vertical
    /// NPN's.
    sign: f64,
    /// The saturation current and exponential slope of the junction's diode, if it has one.
    diode: Option<(R, R)>,
    /// The thermal voltage at the transistor's temperature.
    thermal_voltage: f64,
    critical_voltage: f64,
    /// The depletion, whose capacitance grows linearly from zero bias.
    depletion: Option<Depletion<R>>,
}

/// A transistor's substrate junction, between the nodes it joins.
#[derive(Debug)]
struct Substrate<R: Real> {
    /// The substrate, then the internal collector or base.
    stamp: Stamp<2>,
    junction: SubstrateJunction<R>,
    /// The junction voltage of the last evaluation, from which the next step is limited.
    last: f64,
}

/// The critical voltage of a substrate junction without a diode, from which ngspice limits
/// its steps all the same.
const DIODELESS_CRITICAL_VOLTAGE: f64 = 50.0;

impl<R: Real> Substrate<R> {
    /// Adds the junction's current and its charge.
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let (stamp, junction) = (&self.stamp, &self.junction);
        let applied = junction.sign * (stamp.voltage(input, 0) - stamp.voltage(input, 1));
        let limited = limit_junction(
            applied,
            self.last,
            junction.thermal_voltage,
            junction.critical_voltage,
        )
        .filter(|_| input.limiting);
        if limited.is_some() {
            output.mark_limited();
        }
        let voltage = limited.unwrap_or(applied);
        self.last = voltage;
        let beyond = applied - voltage;
        let zero = R::constant(0.0);
        let (current, conductance) = junction.diode.map_or((zero, zero), |(saturation, slope)| {
            junction_current(voltage, saturation, slope)
        });
        let (current, conductance) = (current + input.gmin * voltage, conductance + input.gmin);
        let pair = (0, 1);
        stamp.current(
            output,
            0,
            1,
            (current + conductance * beyond) * junction.sign,
        );
        stamp.conductance(output, pair, pair, conductance);
        if let Some(depletion) = &junction.depletion {
            let (charge, capacitance) = depletion.evaluate(voltage);
            stamp.charge(
                output,
                0,
                1,
                (charge + capacitance * beyond) * junction.sign,
            );
            stamp.capacitance(output, pair, pair, capacitance);
        }
    }
}

impl<R: Real> BipolarTransistorParameters<R> {
    /// The junctions' DC model at `temperatures`, once each of its parameters is checked.
    fn junctions(&self, temperatures: Temperatures<R>) -> Result<Junctions<R>, ParameterError> {
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
        non_negative("energy_gap", self.energy_gap)?;
        let vt = temperatures.thermal_voltage();
        let scaling = |emission| {
            temperatures.saturation(
                self.energy_gap,
                self.saturation_current_temperature_exponent,
                emission,
            )
        };
        let beta_factor = temperatures.ratio().pow(self.beta_temperature_exponent);
        let (emitter_emission, collector_emission) = (
            self.base_emitter_leakage_emission_coefficient,
            self.base_collector_leakage_emission_coefficient,
        );
        Ok(Junctions {
            saturation_current: self.saturation_current * scaling(R::constant(1.0)),
            forward_beta: self.forward_beta * beta_factor,
            reverse_beta: self.reverse_beta * beta_factor,
            forward_slope: self.forward_emission_coefficient * vt,
            reverse_slope: self.reverse_emission_coefficient * vt,
            emitter_leakage: self.base_emitter_leakage_current * scaling(emitter_emission)
                / beta_factor,
            emitter_leakage_slope: emitter_emission * vt,
            collector_leakage: self.base_collector_leakage_current * scaling(collector_emission)
                / beta_factor,
            collector_leakage_slope: collector_emission * vt,
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
    fn base_resistance(&self) -> Result<Option<BaseResistance<R>>, ParameterError> {
        ParameterError::non_negative("base_resistance", self.base_resistance)?;
        let minimum = self.minimum_base_resistance.unwrap_or(self.base_resistance);
        ParameterError::non_negative("minimum_base_resistance", minimum)?;
        if let Some(current) = self.base_resistance_current {
            ParameterError::positive("base_resistance_current", current)?;
        }
        Ok(
            (self.base_resistance.value() > 0.0).then_some(BaseResistance {
                maximum: self.base_resistance,
                minimum,
                halfway_current: self.base_resistance_current,
            }),
        )
    }

    /// What the transistor stores at `temperatures`, if anything, once each of its
    /// parameters is checked.
    fn charges(&self, temperatures: Temperatures<R>) -> Result<Option<Charges<R>>, ParameterError> {
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
        if !(0.0..=1.0).contains(&self.internal_base_fraction.value()) {
            return Err(ParameterError(format!(
                "internal_base_fraction must be at least 0 and at most 1, not {:?}",
                self.internal_base_fraction.value()
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
        .any(|value| value.value() > 0.0);
        let (emitter_potential, emitter_capacitance) = temperatures.junction(
            self.base_emitter_potential,
            self.base_emitter_capacitance,
            self.base_emitter_grading,
        );
        let (collector_potential, collector_capacitance) = temperatures.junction(
            self.base_collector_potential,
            self.base_collector_capacitance,
            self.base_collector_grading,
        );
        let internal = collector_capacitance * self.internal_base_fraction;
        let collector_depletion = |capacitance| {
            Depletion::new(
                capacitance,
                collector_potential,
                self.base_collector_grading,
                self.depletion_coefficient,
            )
        };
        Ok(stores.then(|| Charges {
            base_emitter: Depletion::new(
                emitter_capacitance,
                emitter_potential,
                self.base_emitter_grading,
                self.depletion_coefficient,
            ),
            base_collector: collector_depletion(internal),
            external_base_collector: collector_depletion(collector_capacitance - internal),
            forward_transit_time: self.forward_transit_time,
            reverse_transit_time: self.reverse_transit_time,
            bias_coefficient: self.transit_time_bias_coefficient,
            voltage_factor,
            high_current: self.transit_time_current,
        }))
    }
}

impl<R: Real> BipolarTransistorParameters<R> {
    /// The substrate junction at `temperatures`, if the transistor has a substrate, once
    /// each of its parameters is checked.
    fn substrate_junction(
        &self,
        temperatures: Temperatures<R>,
    ) -> Result<Option<SubstrateJunction<R>>, ParameterError> {
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
        let emission = self
            .substrate_emission_coefficient
            .unwrap_or(R::constant(1.0));
        ParameterError::positive("substrate_emission_coefficient", emission)?;
        let capacitance = self.substrate_capacitance.unwrap_or(R::constant(0.0));
        ParameterError::non_negative("substrate_capacitance", capacitance)?;
        let potential = self.substrate_potential.unwrap_or(R::constant(0.75));
        ParameterError::positive("substrate_potential", potential)?;
        let grading = below_one(
            "substrate_grading",
            self.substrate_grading.unwrap_or(R::constant(0.0)),
        )?;
        let vt = temperatures.thermal_voltage();
        // ngspice scales the substrate's saturation current as the transport current's.
        let scaling = temperatures.saturation(
            self.energy_gap,
            self.saturation_current_temperature_exponent,
            R::constant(1.0),
        );
        let diode = self
            .substrate_saturation_current
            .map(|current| {
                ParameterError::positive("substrate_saturation_current", current)
                    .map(|current| (current * scaling, emission * vt))
            })
            .transpose()?;
        let (potential, capacitance) = temperatures.junction(potential, capacitance, grading);
        let orientation = match construction {
            Construction::Vertical => 1.0,
            Construction::Lateral => -1.0,
        };
        Ok(Some(SubstrateJunction {
            construction,
            sign: self.polarity.sign() * orientation,
            diode,
            thermal_voltage: vt,
            critical_voltage: diode.map_or(DIODELESS_CRITICAL_VOLTAGE, |(current, _)| {
                critical_voltage(current.value(), vt)
            }),
            depletion: Depletion::new(capacitance, potential, grading, R::constant(0.0)),
        }))
    }
}

/// The reciprocal of an optional positive parameter, zero when it is left out, infinite.
fn inverse<R: Real>(name: &str, value: Option<R>) -> Result<R, ParameterError> {
    value.map_or(Ok(R::constant(0.0)), |value| {
        ParameterError::positive(name, value).map(Real::recip)
    })
}

impl NativeModel for BipolarTransistor {
    type Device<R: Real> = BipolarTransistor<R>;
}

impl<R: Real> NativeDevice<R> for BipolarTransistor<R> {
    const NAME: &'static str = "bipolar_transistor";
    type Parameters = BipolarTransistorParameters<R>;

    fn interface(parameters: &BipolarTransistorParameters<R>) -> Interface {
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
        parameters: BipolarTransistorParameters<R>,
        binding: &mut Binding<'_>,
    ) -> Result<Self, ParameterError> {
        let temperatures =
            Temperatures::new(binding.temperature(), parameters.nominal_temperature)?;
        let junctions = parameters.junctions(temperatures)?;
        let base = parameters.base_resistance()?;
        let charges = parameters.charges(temperatures)?;
        let substrate_junction = parameters.substrate_junction(temperatures)?;
        let non_negative = ParameterError::non_negative;
        let collector_resistance =
            non_negative("collector_resistance", parameters.collector_resistance)?;
        let emitter_resistance = non_negative("emitter_resistance", parameters.emitter_resistance)?;

        let [collector, base_terminal, emitter] = [0, 1, 2].map(|index| binding.terminal(index));
        let inner_collector = binding.behind(collector, collector_resistance, "collector");
        let inner_base = binding.behind(base_terminal, parameters.base_resistance, "base");
        let inner_emitter = binding.behind(emitter, emitter_resistance, "emitter");
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
        Ok(Self {
            stamp,
            sign: parameters.polarity.sign(),
            collector_conductance: series_conductance(collector_resistance),
            emitter_conductance: series_conductance(emitter_resistance),
            base,
            thermal_voltage: temperatures.thermal_voltage(),
            critical_voltage: critical_voltage(
                junctions.saturation_current.value(),
                temperatures.thermal_voltage(),
            ),
            junctions,
            charges,
            substrate,
            last: [0.0; 2],
        })
    }
}

/// The base-emitter and base-collector voltages of one evaluation.
type Bias = stamp::Bias<2>;

const BASE_EMITTER: (usize, usize) = (INNER_BASE, INNER_EMITTER);
const BASE_COLLECTOR: (usize, usize) = (INNER_BASE, INNER_COLLECTOR);

impl<R: Real> BipolarTransistor<R> {
    /// Adds the collector and emitter resistances.
    fn add_series(&self, input: &Input<'_>, output: &mut Output<'_, R>) {
        let stamp = &self.stamp;
        stamp.series(
            input,
            output,
            (COLLECTOR, INNER_COLLECTOR),
            self.collector_conductance,
        );
        stamp.series(
            input,
            output,
            (EMITTER, INNER_EMITTER),
            self.emitter_conductance,
        );
    }

    /// The junction voltages at `input`, limited from the last evaluation's.
    fn bias(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) -> Bias {
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
                self.thermal_voltage,
                self.critical_voltage,
            )
            .filter(|_| input.limiting)
        });
        if limits.iter().any(Option::is_some) {
            output.mark_limited();
        }
        let voltages = [0, 1].map(|index| limits[index].unwrap_or(applied[index]));
        self.last = voltages;
        Bias::new(applied, voltages)
    }

    /// Adds the transport current and the base currents through each junction.
    fn add_currents(&self, bias: &Bias, state: &State<R>, output: &mut Output<'_, R>) {
        let (stamp, sign) = (&self.stamp, self.sign);
        let (transport, slopes) = state.transport;
        let collector_emitter = (INNER_COLLECTOR, INNER_EMITTER);
        let current = (transport + bias.extend(slopes)) * sign;
        stamp.current(output, INNER_COLLECTOR, INNER_EMITTER, current);
        stamp.conductance(output, collector_emitter, BASE_EMITTER, slopes[0]);
        stamp.conductance(output, collector_emitter, BASE_COLLECTOR, slopes[1]);
        for (index, pair) in [BASE_EMITTER, BASE_COLLECTOR].into_iter().enumerate() {
            let (current, conductance) = state.base[index];
            let current = (current + conductance * bias.beyond[index]) * sign;
            stamp.current(output, pair.0, pair.1, current);
            stamp.conductance(output, pair, pair, conductance);
        }
    }

    /// Adds the current through the base resistance, `drop / r`, which varies with the
    /// drop and with the junction voltages the resistance depends on.
    fn add_base_resistance(
        &self,
        resistance: &BaseResistance<R>,
        bias: &Bias,
        state: &State<R>,
        input: &Input<'_>,
        output: &mut Output<'_, R>,
    ) {
        let stamp = &self.stamp;
        let (value, slopes) = resistance.evaluate(state);
        let drop = stamp.voltage(input, BASE) - stamp.voltage(input, INNER_BASE);
        let slopes = slopes.map(|slope| slope * -drop / (value * value));
        let pair = (BASE, INNER_BASE);
        stamp.current(
            output,
            BASE,
            INNER_BASE,
            R::constant(drop) / value + bias.extend(slopes),
        );
        stamp.conductance(output, pair, pair, value.recip());
        stamp.conductance(output, pair, BASE_EMITTER, slopes[0] * self.sign);
        stamp.conductance(output, pair, BASE_COLLECTOR, slopes[1] * self.sign);
    }

    /// Adds the charges of the junctions, and the external base-collector depletion,
    /// whose voltage SPICE never limits.
    fn add_charges(
        &self,
        charges: &Charges<R>,
        bias: &Bias,
        state: &State<R>,
        input: &Input<'_>,
        output: &mut Output<'_, R>,
    ) {
        let (stamp, sign) = (&self.stamp, self.sign);
        let [vbe, vbc] = bias.voltages;
        let (charge, capacitance, cross) = charges.base_emitter(vbe, vbc, state);
        let charge = (charge + bias.extend([capacitance, cross])) * sign;
        stamp.charge(output, INNER_BASE, INNER_EMITTER, charge);
        stamp.capacitance(output, BASE_EMITTER, BASE_EMITTER, capacitance);
        stamp.capacitance(output, BASE_EMITTER, BASE_COLLECTOR, cross);
        let (charge, capacitance) = charges.base_collector(vbc, state);
        let charge = (charge + capacitance * bias.beyond[1]) * sign;
        stamp.charge(output, INNER_BASE, INNER_COLLECTOR, charge);
        stamp.capacitance(output, BASE_COLLECTOR, BASE_COLLECTOR, capacitance);
        let external = sign * (stamp.voltage(input, BASE) - stamp.voltage(input, INNER_COLLECTOR));
        let (charge, capacitance) =
            depletion_charge(charges.external_base_collector.as_ref(), external);
        let pair = (BASE, INNER_COLLECTOR);
        stamp.charge(output, BASE, INNER_COLLECTOR, charge * sign);
        stamp.capacitance(output, pair, pair, capacitance);
    }
}

impl<R: Real> Device<R> for BipolarTransistor<R> {
    fn evaluate(&mut self, input: &Input<'_>, output: &mut Output<'_, R>) {
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
    use crate::testing::{Bench, assert_jacobians_match, assert_parameter_gradients_match};
    use crate::thermal_voltage;

    const THERMAL_VOLTAGE: f64 = thermal_voltage(REFERENCE_TEMPERATURE);

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

    #[test]
    fn parameter_gradients_are_exact() {
        let temperature = r#""energy_gap": 1.2, "saturation_current_temperature_exponent": 3.5,
            "beta_temperature_exponent": 1.2, "nominal_temperature": 290, "#;
        let full = FULL.replace('{', &format!("{{{temperature}"));
        let resistive = |halfway: &str| {
            full.replace(
                '{',
                &format!(
                    r#"{{"base_resistance": 200, "minimum_base_resistance": 20, {halfway}
                    "collector_resistance": 10, "emitter_resistance": 2, "#
                ),
            )
        };
        for [collector, base] in [[3.0, 0.7], [0.4, 0.72], [-0.5, 0.1], [0.2, -0.6]] {
            assert_parameter_gradients_match::<BipolarTransistor>(&full, &[collector, base, 0.0]);
            for halfway in ["", r#""base_resistance_current": 1e-4,"#] {
                let substrate = resistive(halfway).replace(
                    '{',
                    r#"{"substrate": "lateral", "substrate_saturation_current": 1e-16,
                        "substrate_emission_coefficient": 1.2, "substrate_capacitance": 3e-12,
                        "substrate_potential": 0.6, "substrate_grading": 0.4, "#,
                );
                assert_parameter_gradients_match::<BipolarTransistor>(
                    &substrate,
                    &[
                        collector,
                        base,
                        0.0,
                        -1.0,
                        collector - 0.01,
                        base - 0.02,
                        0.001,
                    ],
                );
            }
        }
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
