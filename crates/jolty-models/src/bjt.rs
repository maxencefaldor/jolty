#![cfg_attr(
    not(test),
    deny(
        clippy::indexing_slicing,
        clippy::panic,
        reason = "a kernel never panics, and reads fixed arrays through `get`"
    )
)]
//! The Gummel-Poon bipolar transistor, derived from ngspice 47's implementation of it in
//! `src/spicelib/devices/bjt/`:
//! Copyright 1985 - 2018, Regents of the University of California and others. Used under
//! ngspice's Modified BSD license, which the repository's `NOTICE` reproduces.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::junction::{
    Depletion, Temperatures, below_one, critical_voltage, depletion_charge, junction_current,
    limit_junction,
};
use crate::kernel::parts::SMALL_RESISTANCE;
use crate::kernel::parts::{self, CurrentBranch, Nodes, Series, series_value};
use crate::noise::Flicker;
use crate::region::Bipolar;
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Local, Nature, NoiseSource, ParameterError,
    REFERENCE_TEMPERATURE, Real, Regions, Stamp, Unknown,
};

/// The doping order of a bipolar transistor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
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
/// junction current and a leakage one. The transport current from
/// collector to emitter, `(Ibe − Ibc) / qb`, falls with the base charge `qb`, which models
/// the Early effect and high injection. The base resistance falls from `rb` towards `rbm`
/// as the base charge or the base current grows. Each junction holds a depletion charge,
/// and a diffusion charge set by its transit time; the forward transit time grows with
/// the current and the base-collector voltage. Part of the base-collector depletion charge
/// sits on the external base. A base, collector or emitter resistance adds an internal
/// node, labelled `base`, `collector` or `emitter`; one below
/// [`SMALL_RESISTANCE`](crate::SMALL_RESISTANCE), the base's at its minimum, also carries
/// its current as an unknown, labelled `base_current`, `collector_current` or
/// `emitter_current`.
///
/// A transistor with a substrate has a fourth terminal, `substrate`, joined by a junction
/// to its internal collector if it is vertical or to its internal base if it is lateral.
/// The junction conducts the minimum conductance, and its diode's current if it has a
/// saturation current; its depletion capacitance grows linearly in forward bias.
///
/// An excess phase delays the forward transport current, `Ibe / qb`, through Weil's
/// approximation as SPICE makes it, a second-order Bessel filter,
/// `3 / (s² · td² + 3 · s · td + 3)`: its group delay at low frequencies is `td`, and its
/// phase at the frequency `1 / (2π · TF)` nearly the excess phase. Two unknowns carry the
/// filter, labelled `delayed_transport` and `delayed_transport_slope`.
///
/// Away from its nominal temperature, its saturation currents scale with the energy gap,
/// its betas by their temperature exponent, and its junctions' potentials and
/// capacitances with silicon's band gap, as ngspice scales them.
#[derive(Debug, Clone, Copy)]
pub struct BipolarTransistor;

/// Where a transistor's contributions go among its local unknowns: through its collector
/// and emitter resistances and its base resistance, among the base terminal and the
/// collector, base and emitter behind the resistances, and through its excess phase's
/// filter and its substrate junction, if it has them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BipolarTransistorShape {
    stamp: Nodes<4>,
    collector_series: Series,
    emitter_series: Series,
    base: Option<BaseForm>,
    excess: Option<ExcessPhase>,
    /// The substrate, then the internal collector or base.
    substrate: Option<Nodes<2>>,
}

/// A transistor's numbers at its temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedBipolarTransistor<R: Real> {
    sign: f64,
    /// The collector and the emitter resistances' conductances, or each resistance where
    /// its current is an unknown.
    series: [R; 2],
    base: Option<BaseResistance<R>>,
    junctions: Junctions<R>,
    charges: Option<Charges<R>>,
    /// The delay of the excess phase, in seconds.
    delay: R,
    substrate: Option<SubstrateJunction<R>>,
    /// The thermal voltage at the transistor's temperature.
    thermal_voltage: f64,
    critical_voltage: f64,
    temperature: f64,
    flicker: Flicker,
}

/// What a transistor remembers.
#[derive(Debug, Clone, Copy, Default)]
pub struct BipolarTransistorMemory {
    /// The base-emitter and base-collector voltages of the last evaluation, from which the
    /// next steps are limited.
    last: [f64; 2],
    /// The substrate junction's voltage of the last evaluation.
    substrate: f64,
    /// The values of [`Bipolar::VARIABLE_NAMES`] at the last evaluation.
    variables: [f64; 3],
}

/// The indices of the terminals a substrate junction may join: the collector and the base.
const COLLECTOR: usize = 0;
const BASE: usize = 1;

/// Local node indices: the base terminal, then the collector, base and emitter behind their
/// resistances, which are the terminals themselves without one.
const OUTER_BASE: usize = 0;
const INNER_COLLECTOR: usize = 1;
const INNER_BASE: usize = 2;
const INNER_EMITTER: usize = 3;

/// The parameters of a [`BipolarTransistor`], with SPICE's defaults. A parameter SPICE
/// treats as infinite at zero, such as an Early voltage, is infinite when left out.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
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
    /// The excess phase at the frequency `1 / (2π · TF)`, `PTF`, in degrees, which delays
    /// the forward transport current by `PTF · π / 180 · TF`; default 0.
    pub excess_phase: R,
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
    /// The flicker noise coefficient of the base current, `KF`; default 0, no flicker
    /// noise.
    pub flicker_coefficient: R,
    /// The power of the base current in the flicker noise, `AF`; default 1.
    pub flicker_exponent: R,
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
            excess_phase: R::constant(0.0),
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
            flicker_coefficient: R::constant(0.0),
            flicker_exponent: R::constant(1.0),
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
    /// Its forward part, `Ibe / qb`, which an excess phase delays, and its derivatives.
    forward_transport: (R, [R; 2]),
    /// Its reverse part, `Ibc / qb`, which it leaves, and its derivatives: computed apart,
    /// not as the difference of the two above, which cancel.
    reverse_transport: (R, [R; 2]),
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
        let forward_transport = (
            forward.0 / qb,
            [
                (forward.1 - forward.0 * dqb[0] / qb) / qb,
                -forward.0 * dqb[1] / (qb * qb),
            ],
        );
        let reverse_transport = (
            reverse.0 / qb,
            [
                -reverse.0 * dqb[0] / (qb * qb),
                (reverse.1 - reverse.0 * dqb[1] / qb) / qb,
            ],
        );
        State {
            ideal: [forward, reverse],
            base,
            transport,
            forward_transport,
            reverse_transport,
            charge: (qb, dqb),
        }
    }
}

/// How the base resistance joins the base terminal to the internal base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BaseForm {
    /// A conductance between the two.
    Conductance,
    /// With a minimum below [`SMALL_RESISTANCE`], the base current as a local unknown of
    /// its own, held by `drop − r · i = 0`.
    Current(BaseCurrent),
}

/// The base current as a local unknown of its own, and the local entries of its equation
/// at itself and at the internal base, emitter and collector, on whose voltages the
/// resistance depends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BaseCurrent {
    branch: CurrentBranch,
    entries: [usize; 4],
}

impl BaseCurrent {
    /// The current from the `base` terminal to the internal one, labelled `base_current`,
    /// across a resistance that depends on the voltages of the internal `emitter` and
    /// `collector`.
    fn new(layout: &mut Layout, base: [usize; 2], emitter: usize, collector: usize) -> Self {
        let branch = CurrentBranch::new(layout, "base_current", base);
        let row = branch.unknown();
        let [_, inner] = base;
        let entries = [row, inner, emitter, collector].map(|column| layout.entry(row, column));
        Self { branch, entries }
    }

    /// Adds the current's equation for a `resistance` with derivatives `slopes` with respect
    /// to the base-emitter and base-collector voltages of a transistor of polarity `sign`.
    fn add<R: Real>(
        &self,
        (resistance, slopes): (R, [R; 2]),
        (bias, sign): (&Bias, f64),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let current = self.branch.current(at);
        let drop = R::constant(self.branch.voltage(at));
        let residual = drop - (resistance + bias.extend(slopes)) * current;
        self.branch.add(at, stamp, residual);
        let [emitter, collector] = slopes.map(|slope| slope * (current * sign));
        let [own, at_base, at_emitter, at_collector] = self.entries;
        stamp.add_conductance(own, -resistance);
        stamp.add_conductance(at_base, -(emitter + collector));
        stamp.add_conductance(at_emitter, emitter);
        stamp.add_conductance(at_collector, collector);
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

/// The forward transport current `u` delayed through the filter of an excess phase: the
/// delayed current `x` and `y = td · dx/dt`, both in amperes, held by `td · dx/dt − y = 0`
/// and `td · dy/dt + 3 · y + 3 · x − 3 · u = 0`, so `td² · x'' + 3 · td · x' + 3 · x = 3 · u`.
/// At the operating point `x` is `u`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExcessPhase {
    /// The local unknowns `x` and `y`.
    unknowns: [usize; 2],
    /// The local entries of `x`'s equation at `x` and `y`, and of `y`'s at `x` and `y`.
    own: [[usize; 2]; 2],
    /// The local entries of `y`'s equation at the internal base, emitter and collector.
    driven: [usize; 3],
    /// The local entries of the internal collector's and emitter's equations at `x`.
    delivered: [usize; 2],
}

impl ExcessPhase {
    /// The filter between the internal `nodes`: base, emitter and collector.
    fn new(layout: &mut Layout, [base, emitter, collector]: [usize; 3]) -> Self {
        let unknowns = [
            layout.internal("delayed_transport", Nature::Current),
            layout.internal("delayed_transport_slope", Nature::Current),
        ];
        let [x, y] = unknowns;
        Self {
            unknowns,
            own: [x, y].map(|row| [x, y].map(|column| layout.entry(row, column))),
            driven: [base, emitter, collector].map(|column| layout.entry(y, column)),
            delivered: [collector, emitter].map(|row| layout.entry(row, x)),
        }
    }

    /// The delayed forward transport current, `x`.
    fn delayed(&self, at: &Local<'_>) -> f64 {
        let [x, _] = self.unknowns;
        at.value(x)
    }

    /// Adds the filter's two equations, of `delay`, for the forward transport current of
    /// `state`, of a transistor of polarity `sign`.
    fn add<R: Real>(
        &self,
        (bias, state): (&Bias, &State<R>),
        (delay, sign): (R, f64),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [x, y] = self.unknowns;
        let [delayed, slope] = [x, y].map(|unknown| at.value(unknown));
        let (forward, slopes) = state.forward_transport;
        let driven = forward + bias.extend(slopes);
        stamp.add_residual(x, R::constant(-slope));
        stamp.add_charge(x, delay * delayed);
        stamp.add_residual(y, (R::constant(delayed + slope) - driven) * 3.0);
        stamp.add_charge(y, delay * slope);
        let [[x_x, x_y], [y_x, y_y]] = self.own;
        stamp.add_conductance(x_y, R::constant(-1.0));
        stamp.add_capacitance(x_x, delay);
        stamp.add_conductance(y_x, R::constant(3.0));
        stamp.add_conductance(y_y, R::constant(3.0));
        stamp.add_capacitance(y_y, delay);
        let [at_base, at_emitter, at_collector] = self.driven;
        let [by_emitter, by_collector] = slopes;
        stamp.add_conductance(at_base, -(by_emitter + by_collector) * (3.0 * sign));
        stamp.add_conductance(at_emitter, by_emitter * (3.0 * sign));
        stamp.add_conductance(at_collector, by_collector * (3.0 * sign));
    }
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

/// The critical voltage of a substrate junction without a diode, from which ngspice limits
/// its steps all the same.
const DIODELESS_CRITICAL_VOLTAGE: f64 = 50.0;

impl<R: Real> SubstrateJunction<R> {
    /// Adds the junction's current and its charge, between the substrate and the internal
    /// node it joins, `nodes`, at `at`, its last voltage `last`, which it updates.
    fn evaluate(&self, nodes: &Nodes<2>, last: &mut f64, at: &Local<'_>, stamp: &mut Stamp<'_, R>) {
        let applied = self.sign * (nodes.voltage(at, 0) - nodes.voltage(at, 1));
        let limited = limit_junction(applied, *last, self.thermal_voltage, self.critical_voltage)
            .filter(|_| at.limiting);
        if limited.is_some() {
            stamp.mark_limited();
        }
        let voltage = limited.unwrap_or(applied);
        *last = voltage;
        let beyond = applied - voltage;
        let zero = R::constant(0.0);
        let (current, conductance) = self.diode.map_or((zero, zero), |(saturation, slope)| {
            junction_current(voltage, saturation, slope)
        });
        let (current, conductance) = (current + at.gmin * voltage, conductance + at.gmin);
        let pair = (0, 1);
        nodes.current(stamp, 0, 1, (current + conductance * beyond) * self.sign);
        nodes.conductance(stamp, pair, pair, conductance);
        if let Some(depletion) = &self.depletion {
            let (charge, capacitance) = depletion.evaluate(voltage);
            nodes.charge(stamp, 0, 1, (charge + capacitance * beyond) * self.sign);
            nodes.capacitance(stamp, pair, pair, capacitance);
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
            sign: self.polarity.sign() * orientation,
            diode,
            thermal_voltage: vt.value(),
            critical_voltage: diode.map_or(DIODELESS_CRITICAL_VOLTAGE, |(current, _)| {
                critical_voltage(current.value(), vt.value())
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

impl Kernel for BipolarTransistor {
    const NAME: &'static str = "bipolar_transistor";
    type Parameters<R: Real> = BipolarTransistorParameters<R>;
    type Shape = BipolarTransistorShape;
    type Prepared<R: Real> = PreparedBipolarTransistor<R>;
    type Memory = BipolarTransistorMemory;

    fn interface<R: Real>(parameters: &BipolarTransistorParameters<R>) -> Interface {
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

    fn prepare<R: Real>(
        parameters: &BipolarTransistorParameters<R>,
        temperature: R,
    ) -> Result<PreparedBipolarTransistor<R>, ParameterError> {
        let Scaled {
            junctions,
            charges,
            substrate,
            thermal_voltage,
            critical_voltage,
            temperature,
        } = parameters.scaled(temperature)?;
        let flicker = Flicker::new(parameters.flicker_coefficient, parameters.flicker_exponent)?;
        let base = parameters.base_resistance()?;
        let delay = parameters.excess_phase_delay()?;
        let [collector_resistance, emitter_resistance] = parameters.series_resistances()?;
        Ok(PreparedBipolarTransistor {
            sign: parameters.polarity.sign(),
            series: [
                series_value(collector_resistance),
                series_value(emitter_resistance),
            ],
            base,
            junctions,
            charges,
            delay,
            substrate,
            thermal_voltage,
            critical_voltage,
            temperature,
            flicker,
        })
    }

    fn layout<R: Real>(
        parameters: &BipolarTransistorParameters<R>,
        _: &PreparedBipolarTransistor<R>,
        layout: &mut Layout,
    ) -> Result<BipolarTransistorShape, ParameterError> {
        let base_resistance = parameters.base_resistance()?;
        let delay = parameters.excess_phase_delay()?;
        let [collector_resistance, emitter_resistance] = parameters.series_resistances()?;
        let [collector, base_terminal, emitter] = [0, 1, 2];
        let (collector_series, inner_collector) = Series::behind(
            layout,
            collector,
            collector_resistance,
            ("collector", "collector_current"),
        );
        let inner_base = if parameters.base_resistance.value() > 0.0 {
            layout.internal("base", Nature::Voltage)
        } else {
            base_terminal
        };
        let (emitter_series, inner_emitter) = Series::behind(
            layout,
            emitter,
            emitter_resistance,
            ("emitter", "emitter_current"),
        );
        let substrate = parameters.substrate.map(|construction| {
            let joined = match construction {
                Construction::Vertical => inner_collector,
                Construction::Lateral => inner_base,
            };
            Nodes::new(layout, [3, joined])
        });
        let base = base_resistance.map(|resistance| {
            if resistance.minimum.value() < SMALL_RESISTANCE {
                let nodes = [base_terminal, inner_base];
                BaseForm::Current(BaseCurrent::new(
                    layout,
                    nodes,
                    inner_emitter,
                    inner_collector,
                ))
            } else {
                BaseForm::Conductance
            }
        });
        let stamp = Nodes::new(
            layout,
            [base_terminal, inner_collector, inner_base, inner_emitter],
        );
        let excess = (delay.value() > 0.0)
            .then(|| ExcessPhase::new(layout, [inner_base, inner_emitter, inner_collector]));
        Ok(BipolarTransistorShape {
            stamp,
            collector_series,
            emitter_series,
            base,
            excess,
            substrate,
        })
    }

    fn evaluate<R: Real>(
        shape: &BipolarTransistorShape,
        prepared: &PreparedBipolarTransistor<R>,
        memory: &mut BipolarTransistorMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [collector_series, emitter_series] = prepared.series;
        shape.collector_series.add(collector_series, at, stamp);
        shape.emitter_series.add(emitter_series, at, stamp);
        let bias = prepared.bias(shape, memory, at, stamp);
        let [vbe, vbc] = bias.voltages;
        memory.variables = Bipolar::variables(vbe, vbc, prepared.sign);
        let state = prepared.junctions.evaluate(vbe, vbc, at.gmin);
        let added = Added {
            shape,
            prepared,
            bias: &bias,
            state: &state,
        };
        added.currents(at, stamp);
        if let Some(excess) = &shape.excess {
            excess.add((&bias, &state), (prepared.delay, prepared.sign), at, stamp);
        }
        if let (Some(resistance), Some(form)) = (&prepared.base, &shape.base) {
            added.base_resistance((resistance, form), at, stamp);
        }
        if let Some(charges) = &prepared.charges {
            added.charges(charges, at, stamp);
        }
        if let (Some(junction), Some(nodes)) = (&prepared.substrate, &shape.substrate) {
            junction.evaluate(nodes, &mut memory.substrate, at, stamp);
        }
    }

    fn variable_names() -> &'static [&'static str] {
        &Bipolar::VARIABLE_NAMES
    }

    fn variables(memory: &BipolarTransistorMemory, values: &mut [f64]) {
        parts::write_variables(&memory.variables, values);
    }

    fn regions() -> Regions {
        Bipolar::REGIONS
    }

    /// The thermal noise of the series resistances, and the shot noise of the collector
    /// and base currents, with the base current's flicker noise, as SPICE models them.
    fn noise<R: Real>(
        shape: &BipolarTransistorShape,
        prepared: &PreparedBipolarTransistor<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let temperature = prepared.temperature;
        let [collector_series, emitter_series] = prepared.series;
        shape.collector_series.noise(
            collector_series,
            ("collector thermal", temperature),
            sources,
        );
        shape
            .emitter_series
            .noise(emitter_series, ("emitter thermal", temperature), sources);
        let nodes = &shape.stamp;
        let base = nodes.voltage(at, INNER_BASE);
        let vbe = prepared.sign * (base - nodes.voltage(at, INNER_EMITTER));
        let vbc = prepared.sign * (base - nodes.voltage(at, INNER_COLLECTOR));
        let state = prepared.junctions.evaluate(vbe, vbc, at.gmin);
        let node = |index: usize| nodes.equation(index);
        if let (Some(resistance), Some(form)) = (&prepared.base, &shape.base) {
            let resistance = resistance.evaluate(&state).0.value();
            sources.push(match form {
                BaseForm::Conductance => NoiseSource::thermal(
                    "base thermal",
                    [node(OUTER_BASE), node(INNER_BASE)],
                    resistance.recip(),
                    temperature,
                ),
                BaseForm::Current(current) => NoiseSource::thermal_in_series(
                    "base thermal",
                    Unknown::new(current.branch.unknown()),
                    resistance,
                    temperature,
                ),
            });
        }
        let [(base_emitter, _), (base_collector, _)] = state.base;
        let collector = state.transport.0 - base_collector;
        let base_current = (base_emitter + base_collector).value();
        sources.push(NoiseSource::shot(
            "collector shot",
            [node(INNER_COLLECTOR), node(INNER_EMITTER)],
            collector.value(),
        ));
        let base_to_emitter = [node(INNER_BASE), node(INNER_EMITTER)];
        sources.push(NoiseSource::shot(
            "base shot",
            base_to_emitter,
            base_current,
        ));
        prepared.flicker.add(base_to_emitter, base_current, sources);
    }
}

/// What a transistor's temperature sets.
struct Scaled<R: Real> {
    junctions: Junctions<R>,
    charges: Option<Charges<R>>,
    substrate: Option<SubstrateJunction<R>>,
    thermal_voltage: f64,
    critical_voltage: f64,
    temperature: f64,
}

impl<R: Real> BipolarTransistorParameters<R> {
    /// What the transistor's `temperature`, in kelvins, sets, once each parameter it
    /// reads is checked.
    fn scaled(&self, temperature: R) -> Result<Scaled<R>, ParameterError> {
        let temperatures = Temperatures::of_junction(temperature, self.nominal_temperature)?;
        let junctions = self.junctions(temperatures)?;
        let thermal_voltage = temperatures.thermal_voltage().value();
        Ok(Scaled {
            critical_voltage: critical_voltage(
                junctions.saturation_current.value(),
                thermal_voltage,
            ),
            junctions,
            charges: self.charges(temperatures)?,
            substrate: self.substrate_junction(temperatures)?,
            thermal_voltage,
            temperature: temperatures.actual.value(),
        })
    }
}

impl<R: Real> BipolarTransistorParameters<R> {
    /// The delay of the excess phase, `excess_phase · π / 180 · forward_transit_time`,
    /// once the excess phase is checked.
    fn excess_phase_delay(&self) -> Result<R, ParameterError> {
        let excess_phase = ParameterError::non_negative("excess_phase", self.excess_phase)?;
        Ok(excess_phase * (std::f64::consts::PI / 180.0) * self.forward_transit_time)
    }

    /// The collector and emitter resistances, once each is checked.
    fn series_resistances(&self) -> Result<[R; 2], ParameterError> {
        let non_negative = ParameterError::non_negative;
        Ok([
            non_negative("collector_resistance", self.collector_resistance)?,
            non_negative("emitter_resistance", self.emitter_resistance)?,
        ])
    }
}

/// The base-emitter and base-collector voltages of one evaluation.
type Bias = parts::Bias<2>;

const BASE_EMITTER: (usize, usize) = (INNER_BASE, INNER_EMITTER);
const BASE_COLLECTOR: (usize, usize) = (INNER_BASE, INNER_COLLECTOR);

impl<R: Real> PreparedBipolarTransistor<R> {
    /// The junction voltages at `at`, limited from the last evaluation's, which `memory`
    /// keeps and takes the new ones.
    fn bias(
        &self,
        shape: &BipolarTransistorShape,
        memory: &mut BipolarTransistorMemory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) -> Bias {
        let nodes = &shape.stamp;
        let base = nodes.voltage(at, INNER_BASE);
        let applied = [
            self.sign * (base - nodes.voltage(at, INNER_EMITTER)),
            self.sign * (base - nodes.voltage(at, INNER_COLLECTOR)),
        ];
        let limit = |applied: f64, last: f64| {
            limit_junction(applied, last, self.thermal_voltage, self.critical_voltage)
                .filter(|_| at.limiting)
        };
        let [
            [applied_emitter, applied_collector],
            [last_emitter, last_collector],
        ] = [applied, memory.last];
        let limits = [
            limit(applied_emitter, last_emitter),
            limit(applied_collector, last_collector),
        ];
        if limits.iter().any(Option::is_some) {
            stamp.mark_limited();
        }
        let [limit_emitter, limit_collector] = limits;
        let voltages = [
            limit_emitter.unwrap_or(applied_emitter),
            limit_collector.unwrap_or(applied_collector),
        ];
        memory.last = voltages;
        Bias::new(applied, voltages)
    }
}

/// How a transistor adds what it computes at one bias, the junctions' `state` there.
struct Added<'a, R: Real> {
    shape: &'a BipolarTransistorShape,
    prepared: &'a PreparedBipolarTransistor<R>,
    bias: &'a Bias,
    state: &'a State<R>,
}

impl<R: Real> Added<'_, R> {
    /// Adds the transport current and the base currents through each junction. With an
    /// excess phase, the delayed forward transport current stands in for the forward one.
    fn currents(&self, at: &Local<'_>, stamp: &mut Stamp<'_, R>) {
        let (nodes, sign) = (&self.shape.stamp, self.prepared.sign);
        let (bias, state) = (self.bias, self.state);
        let (mut transport, mut slopes) = state.transport;
        if let Some(excess) = &self.shape.excess {
            let (reverse, reverse_slopes) = state.reverse_transport;
            transport = R::constant(excess.delayed(at)) - reverse;
            slopes = reverse_slopes.map(|slope| -slope);
            let [at_collector, at_emitter] = excess.delivered;
            stamp.add_conductance(at_collector, R::constant(sign));
            stamp.add_conductance(at_emitter, R::constant(-sign));
        }
        let collector_emitter = (INNER_COLLECTOR, INNER_EMITTER);
        let current = (transport + bias.extend(slopes)) * sign;
        nodes.current(stamp, INNER_COLLECTOR, INNER_EMITTER, current);
        let [by_emitter, by_collector] = slopes;
        nodes.conductance(stamp, collector_emitter, BASE_EMITTER, by_emitter);
        nodes.conductance(stamp, collector_emitter, BASE_COLLECTOR, by_collector);
        let junctions = [BASE_EMITTER, BASE_COLLECTOR].into_iter().zip(state.base);
        for ((pair, (current, conductance)), beyond) in junctions.zip(bias.beyond) {
            let current = (current + conductance * beyond) * sign;
            nodes.current(stamp, pair.0, pair.1, current);
            nodes.conductance(stamp, pair, pair, conductance);
        }
    }

    /// Adds the current through the base resistance, `drop / r`, or the equation of the
    /// base current that crosses it; either varies with the junction voltages the
    /// resistance depends on.
    fn base_resistance(
        &self,
        (resistance, form): (&BaseResistance<R>, &BaseForm),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let (value, slopes) = resistance.evaluate(self.state);
        let sign = self.prepared.sign;
        if let BaseForm::Current(current) = form {
            current.add((value, slopes), (self.bias, sign), at, stamp);
            return;
        }
        let nodes = &self.shape.stamp;
        let drop = nodes.voltage(at, OUTER_BASE) - nodes.voltage(at, INNER_BASE);
        let slopes = slopes.map(|slope| slope * -drop / (value * value));
        let pair = (OUTER_BASE, INNER_BASE);
        nodes.current(
            stamp,
            OUTER_BASE,
            INNER_BASE,
            R::constant(drop) / value + self.bias.extend(slopes),
        );
        nodes.conductance(stamp, pair, pair, value.recip());
        let [by_emitter, by_collector] = slopes;
        nodes.conductance(stamp, pair, BASE_EMITTER, by_emitter * sign);
        nodes.conductance(stamp, pair, BASE_COLLECTOR, by_collector * sign);
    }

    /// Adds the charges of the junctions, and the external base-collector depletion,
    /// whose voltage SPICE never limits.
    fn charges(&self, charges: &Charges<R>, at: &Local<'_>, stamp: &mut Stamp<'_, R>) {
        let (nodes, sign) = (&self.shape.stamp, self.prepared.sign);
        let (bias, state) = (self.bias, self.state);
        let [vbe, vbc] = bias.voltages;
        let (charge, capacitance, cross) = charges.base_emitter(vbe, vbc, state);
        let charge = (charge + bias.extend([capacitance, cross])) * sign;
        nodes.charge(stamp, INNER_BASE, INNER_EMITTER, charge);
        nodes.capacitance(stamp, BASE_EMITTER, BASE_EMITTER, capacitance);
        nodes.capacitance(stamp, BASE_EMITTER, BASE_COLLECTOR, cross);
        let (charge, capacitance) = charges.base_collector(vbc, state);
        let [_, beyond_collector] = bias.beyond;
        let charge = (charge + capacitance * beyond_collector) * sign;
        nodes.charge(stamp, INNER_BASE, INNER_COLLECTOR, charge);
        nodes.capacitance(stamp, BASE_COLLECTOR, BASE_COLLECTOR, capacitance);
        let external = sign * (nodes.voltage(at, OUTER_BASE) - nodes.voltage(at, INNER_COLLECTOR));
        let (charge, capacitance) =
            depletion_charge(charges.external_base_collector.as_ref(), external);
        let pair = (OUTER_BASE, INNER_COLLECTOR);
        nodes.charge(stamp, OUTER_BASE, INNER_COLLECTOR, charge * sign);
        nodes.capacitance(stamp, pair, pair, capacitance);
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::Input;
    use crate::region::{BASE_COLLECTOR, BASE_EMITTER};
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_parameter_gradients_match,
        assert_switches_where_margin_vanishes, source, thermal,
    };
    use crate::{ELEMENTARY_CHARGE, Unknown, thermal_voltage};

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
    fn small_series_resistances_carry_their_currents_as_unknowns() {
        let small = |polarity: &str, halfway: &str| {
            FULL.replace(
                '{',
                &format!(
                    r#"{{"polarity": "{polarity}", "base_resistance": 0.8,
                    "minimum_base_resistance": 0.2, {halfway} "collector_resistance": 0.5,
                    "emitter_resistance": 0.1, "#
                ),
            )
        };
        let bench = Bench::new::<BipolarTransistor>(&small("npn", ""));
        assert_eq!(
            bench.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "device:collector",
                "device:collector_current",
                "device:base",
                "device:emitter",
                "device:emitter_current",
                "device:base_current",
            ]
        );
        for (polarity, sign) in [("npn", 1.0), ("pnp", -1.0)] {
            for halfway in ["", r#""base_resistance_current": 1e-4,"#] {
                let parameters = small(polarity, halfway);
                for [collector, base] in [[3.0, 0.7], [0.4, 0.72], [-0.5, 0.1], [0.2, -0.6]] {
                    // Each current near what its resistance carries at these voltages, as
                    // near a solution: arbitrary currents can leave a node's residual a
                    // near-cancellation that finite differences cannot resolve.
                    let x = [
                        collector,
                        base,
                        0.0,
                        0.98 * collector,
                        0.02 * collector / 0.5,
                        0.97 * base,
                        0.01,
                        -0.01 / 0.1,
                        0.03 * base / 0.5,
                    ]
                    .map(|value| sign * value);
                    assert_jacobians_match::<BipolarTransistor>(&parameters, &x);
                    assert_parameter_gradients_match::<BipolarTransistor>(&parameters, &x);
                }
            }
        }
    }

    /// [`FULL`] with an excess phase of 30°, a delay of 30 · π / 180 · 0.4 ns.
    fn excess() -> String {
        FULL.replace('{', r#"{"excess_phase": 30, "#)
    }

    #[test]
    fn an_excess_phase_delays_the_forward_transport_current_through_two_unknowns() {
        let mut plain = Bench::new::<BipolarTransistor>(FULL);
        let mut delayed = Bench::new::<BipolarTransistor>(&excess());
        assert_eq!(
            delayed.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "device:delayed_transport",
                "device:delayed_transport_slope"
            ]
        );
        let at = [3.0, 0.7, 0.0];
        let rest = plain.settle(&at);
        // y's residual is 3 (x + y − u), so at x = y = 0 it gives the forward transport
        // current u, `Ibe / qb`; at rest x is u, and the transistor's currents are those
        // it has without the delay.
        let forward = -delayed.settle(&[at[0], at[1], at[2], 0.0, 0.0]).residual[4] / 3.0;
        let settled = delayed.settle(&[at[0], at[1], at[2], forward, 0.0]);
        assert_close(&settled.residual[..3], &rest.residual);
        assert!(settled.residual[3].abs() < 1e-30);
        assert!(
            settled.residual[4].abs() < 1e-15 * forward.abs(),
            "{} vs {forward}",
            settled.residual[4]
        );
        // The delayed current's equations hold charges of the delay times each unknown.
        let delay = 30.0 * std::f64::consts::PI / 180.0 * 4e-10;
        assert_close(&settled.charge[3..], &[delay * forward, 0.0]);
        assert_close(
            &[settled.capacitance[3][3], settled.capacitance[4][4]],
            &[delay, delay],
        );
    }

    #[test]
    fn its_delay_is_a_second_order_bessel_filter() {
        // Linearized, the delayed current responds to the forward transport current
        // through 3 / (s² td² + 3 s td + 3): solve the two rows at s = jω for a unit drive.
        let mut delayed = Bench::new::<BipolarTransistor>(&excess());
        let evaluation = delayed.settle(&[3.0, 0.7, 0.0, 1e-3, 0.0]);
        let delay = 30.0 * std::f64::consts::PI / 180.0 * 4e-10;
        for omega in [1e8, 1.0 / delay, 1e10] {
            let entry = |row: usize, column: usize| {
                (
                    evaluation.conductance[row][column],
                    omega * evaluation.capacitance[row][column],
                )
            };
            // The rows of x and y over the columns x and y, and y's drive by the forward
            // transport current, whose conductance at the internal base is −3 du/dvb.
            let [a, b, c, d] = [entry(3, 3), entry(3, 4), entry(4, 3), entry(4, 4)];
            let multiply =
                |p: (f64, f64), q: (f64, f64)| (p.0 * q.0 - p.1 * q.1, p.0 * q.1 + p.1 * q.0);
            let determinant = {
                let (ad, bc) = (multiply(a, d), multiply(b, c));
                (ad.0 - bc.0, ad.1 - bc.1)
            };
            // x = −b · (−3 u) / det for a drive u of 1 into y's row, as y's residual holds −3 u.
            let numerator = multiply(b, (3.0, 0.0));
            let numerator = (-numerator.0, -numerator.1);
            let norm = determinant.0 * determinant.0 + determinant.1 * determinant.1;
            let response = (
                (numerator.0 * determinant.0 + numerator.1 * determinant.1) / norm,
                (numerator.1 * determinant.0 - numerator.0 * determinant.1) / norm,
            );
            let wt = omega * delay;
            let denominator = (3.0 - wt * wt, 3.0 * wt);
            let dnorm = denominator.0 * denominator.0 + denominator.1 * denominator.1;
            let expected = (3.0 * denominator.0 / dnorm, -3.0 * denominator.1 / dnorm);
            assert!(
                (response.0 - expected.0).abs() < 1e-12 && (response.1 - expected.1).abs() < 1e-12,
                "{response:?} vs {expected:?} at {omega}"
            );
        }
    }

    #[test]
    fn an_excess_phase_needs_a_transit_time_and_is_not_negative() {
        let without = Bench::new::<BipolarTransistor>(r#"{"excess_phase": 30}"#);
        assert_eq!(without.unknowns(), ["n0", "n1", "n2"]);
        let error = Bench::try_new::<BipolarTransistor>(r#"{"excess_phase": -1}"#).unwrap_err();
        assert_eq!(
            error.to_string(),
            "excess_phase must be at least 0, not -1.0"
        );
    }

    #[test]
    fn a_heated_transistor_is_the_transistor_at_its_temperature() {
        let model = crate::Thermal::<BipolarTransistor>::new();
        let heated = FULL.replace('{', r#"{"thermal": true, "base_resistance": 50, "#);
        let mut bench = Bench::of(&model, &heated, REFERENCE_TEMPERATURE).expect("valid");
        let cool = FULL.replace('{', r#"{"base_resistance": 50, "#);
        for rise in [0.0, 35.0] {
            let mut plain = Bench::at::<BipolarTransistor>(&cool, REFERENCE_TEMPERATURE + rise);
            let hot = bench.settle(&[3.0, 0.7, 0.0, rise, 0.69]);
            let reference = plain.settle(&[3.0, 0.7, 0.0, 0.69]);
            let electrical = [0, 1, 2, 4].map(|row| hot.residual[row]);
            assert_close(&electrical, &reference.residual);
        }
        for x in [[3.0, 0.7, 0.0, 20.0, 0.69], [0.3, 0.75, 0.0, 70.0, 0.74]] {
            crate::testing::assert_model_jacobians_match(&model, &heated, &x);
            crate::testing::assert_model_parameter_gradients_match(&model, &heated, &x, None);
        }
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
        let evaluation = bench.evaluate_with(&Input::new(&x, None, gmin));
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
        let delayed = full.replace('{', r#"{"excess_phase": 30, "#);
        for [collector, base] in [[3.0, 0.7], [0.4, 0.72], [-0.5, 0.1], [0.2, -0.6]] {
            assert_parameter_gradients_match::<BipolarTransistor>(&full, &[collector, base, 0.0]);
            assert_parameter_gradients_match::<BipolarTransistor>(
                &delayed,
                &[collector, base, 0.0, 2e-3, -1e-4],
            );
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
            assert_jacobians_match::<BipolarTransistor>(
                &parameters.replace('{', r#"{"excess_phase": 45, "#),
                &[sign * collector, sign * base, 0.0, 1e-3, -2e-4],
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

    #[test]
    fn its_currents_make_shot_and_flicker_noise() {
        let parameters = r#"{"saturation_current": 1e-15, "flicker_coefficient": 1e-14,
            "flicker_exponent": 1.2}"#;
        let mut bench = Bench::new::<BipolarTransistor>(parameters);
        let x = [5.0, 0.65, 0.0];
        let evaluation = bench.settle(&x);
        let (collector, base) = (evaluation.residual[0], evaluation.residual[1]);
        let sources = bench.noise(&x);
        let shot = |current: f64| 2.0 * ELEMENTARY_CHARGE * current;
        let collector_shot = source(&sources, "collector shot");
        assert_close(&[collector_shot.density(1.0)], &[shot(collector)]);
        let base_shot = source(&sources, "base shot");
        assert_close(&[base_shot.density(1.0)], &[shot(base)]);
        let flicker = source(&sources, "flicker");
        assert_eq!(flicker.equations, base_shot.equations);
        assert_close(&[flicker.density(10.0)], &[1e-14 * base.powf(1.2) / 10.0]);
        assert_eq!(sources.len(), 3);

        // The series resistances' thermal noise, the small base resistance's a voltage in
        // series with its current.
        let resistive = r#"{"saturation_current": 1e-15, "collector_resistance": 20,
            "emitter_resistance": 2, "base_resistance": 0.5}"#;
        let bench = Bench::new::<BipolarTransistor>(resistive);
        let unknowns = bench.unknowns();
        let at = |name: &str| unknowns.iter().position(|unknown| *unknown == name);
        let base_current = at("device:base_current").expect("the base current");
        let mut x = vec![0.0; unknowns.len()];
        x[..3].copy_from_slice(&[5.0, 0.65, 0.0]);
        let sources = bench.noise(&x);
        assert_close(
            &[
                source(&sources, "collector thermal").density(1.0),
                source(&sources, "emitter thermal").density(1.0),
                source(&sources, "base thermal").density(1.0),
            ],
            &[thermal(0.05), thermal(0.5), thermal(0.5)],
        );
        assert_eq!(
            source(&sources, "base thermal").equations,
            [Some(Unknown::new(base_current)), None]
        );
    }

    /// A transistor is cut off with neither junction forward, active with its base-emitter
    /// junction alone forward, saturated with both, and reversed with its base-collector
    /// junction alone; its margins are the junctions' voltages, which a PNP's mirror, and
    /// each region switches where its margin vanishes.
    #[test]
    fn reports_its_region_and_margins() {
        let mut npn = Bench::new::<BipolarTransistor>("{}");
        let mut pnp = Bench::new::<BipolarTransistor>(r#"{"polarity": "pnp"}"#);
        // The collector, the base and the emitter.
        for (x, region) in [
            ([5.0, -0.5, 0.0], "cut_off"),
            ([5.0, 0.7, 0.0], "active"),
            ([0.2, 0.7, 0.0], "saturated"),
            ([0.0, 0.7, 5.0], "reversed"),
        ] {
            let [collector, base, emitter] = x;
            let margins = [base - emitter, base - collector];
            npn.settle(&x);
            assert_eq!(npn.region(), (region, margins.to_vec()), "{x:?}");
            pnp.settle(&x.map(|value| -value));
            assert_eq!(
                pnp.region(),
                (region, margins.map(|m| -m).to_vec()),
                "{x:?}"
            );
        }
        let switches = [
            ([5.0, -0.1, 0.0], [5.0, 0.1, 0.0], BASE_EMITTER),
            ([5.0, 0.7, 0.0], [0.2, 0.7, 0.0], BASE_COLLECTOR),
            ([0.0, -0.1, 5.0], [0.0, 0.7, 5.0], BASE_COLLECTOR),
        ];
        for (from, to, margin) in switches {
            assert_switches_where_margin_vanishes(&mut npn, (&from, &to), margin);
        }
    }
}
