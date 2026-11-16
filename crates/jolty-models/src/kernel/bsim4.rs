//! BSIM4.5, the Berkeley short-channel IGFET model of CMOS processes from 130 nm down, as
//! ngspice 47 implements it in `src/spicelib/devices/bsim4v5/`, SPICE's level 14 and 54
//! MOSFET of version 4.5.
//!
//! The equations and their defaults are those of the BSIM4.5.0 manual (X. Xi, M. Dunga
//! and others, UC Berkeley, 2005) as ngspice's `b4v5ld.c`, `b4v5temp.c`, `b4v5set.c`,
//! `b4v5geo.c` and `b4v5check.c` write them, constants and all. They are written once, on
//! numbers carrying their derivatives with respect to the voltages, so the conductances
//! and capacitances are the exact derivatives of the currents and charges computed.
//!
//! The modes are those the version 4.5 cards of the `SKY130` and `GF180MCU` process kits
//! select: `mobmod` 0, `capmod` 2, `diomod` 1, `tempmod` 0, `geomod` 0, `rgatemod` 0,
//! `igcmod` 0, `igbmod` 0, `trnqsmod` 0, `acnqsmod` 0, `wpemod` 0, `rgeomod` 0, and
//! `rbodymod`, `rdsmod`, `permod`, `fnoimod` and `tnoimod` 0 or 1; a card's reader
//! refuses any other by name.
//!
//! This model, here and in its submodules, is derived from BSIM4.5.0, developed by the UC
//! Berkeley BSIM Research Group: Copyright 2005 Regents of the University of California.
//! All rights reserved. It is used under the BSIM4 terms of use, which the repository's
//! `NOTICE` reproduces.

mod channel;
mod charges;
mod parameters;
mod prepare;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use self::channel::Operating;
pub use self::parameters::Bsim4Parameters;
use self::prepare::{External, Junction, PreparedBsim4};
use crate::kernel::bsim3::{
    DELTA, EXP_THRESHOLD, MIN_EXP, N_MINLOG, NGSPICE_BOLTZMANN, NOISE_CHARGE, VT0, critical_voltage,
};
use crate::kernel::dual::Dual;
use crate::kernel::junction::{limit_channel, limit_junction};
use crate::kernel::parts::{self, Derivatives, Nodes, Oriented, Series};
use crate::kernel::region::{OVERDRIVE, SATURATION_MARGIN};
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Local, Nature, NoiseSource, ParameterError,
    REGION, Real, Regions, Stamp,
};

/// The permittivity of vacuum, in farads per meter, as `b4v5temp.c` takes it.
const EPS0: f64 = 8.85418e-12;
/// π as `b4v5set.c` and `b4v5temp.c` write it, to ten digits.
#[expect(
    clippy::approx_constant,
    reason = "b4v5set.c's and b4v5temp.c's own π, which their numbers are computed with"
)]
const PI: f64 = 3.141_592_654;

/// A number and its derivatives with respect to a transistor's gate-source, drain-source
/// and bulk-source voltages.
type D<R> = Dual<R, 3>;

/// A junction's voltage, with its derivative with respect to itself.
type J<R> = Dual<R, 1>;

/// `b4v5set.c`'s fringing capacitance per width of an oxide `toxe` thick, of relative
/// permittivity `epsrox`, `2 · epsrox · ε0 / π · ln(1 + 0.4 µm / toxe)`.
fn fringe<R: Real>(epsrox: R, toxe: R) -> R {
    (R::constant(0.4e-6) / toxe + 1.0).ln() * epsrox * (2.0 * EPS0 / PI)
}

/// The body's resistance network, `rbodymod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum BodyResistance {
    /// `rbodymod` 0: the channel and the junctions share the bulk terminal.
    Zero,
    /// `rbodymod` 1: five resistances join the bulk to a node under the channel and one
    /// under each junction.
    One,
}

impl TryFrom<u8> for BodyResistance {
    type Error = String;

    fn try_from(rbodymod: u8) -> Result<Self, String> {
        match rbodymod {
            0 => Ok(Self::Zero),
            1 => Ok(Self::One),
            _ => Err(format!(
                "jolty's BSIM4 has body_resistance_model 0 or 1, not {rbodymod}"
            )),
        }
    }
}

impl From<BodyResistance> for u8 {
    fn from(rbodymod: BodyResistance) -> Self {
        match rbodymod {
            BodyResistance::Zero => 0,
            BodyResistance::One => 1,
        }
    }
}

/// What a diffusion's perimeter `ps` or `pd` includes, `permod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum Perimeter {
    /// `permod` 0: the perimeter includes the gate's edge.
    Zero,
    /// `permod` 1: the perimeter leaves the gate's edge out.
    One,
}

impl TryFrom<u8> for Perimeter {
    type Error = String;

    fn try_from(permod: u8) -> Result<Self, String> {
        match permod {
            0 => Ok(Self::Zero),
            1 => Ok(Self::One),
            _ => Err(format!("BSIM4's perimeter_model is 0 or 1, not {permod}")),
        }
    }
}

impl From<Perimeter> for u8 {
    fn from(permod: Perimeter) -> Self {
        match permod {
            Perimeter::Zero => 0,
            Perimeter::One => 1,
        }
    }
}

/// The source and drain resistances, `rdsmod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum DrainSourceResistance {
    /// `rdsmod` 0: `rdsw` within the drain current, and `rsh` behind the drain and the
    /// source where `nrd` or `nrs` sets one.
    Zero,
    /// `rdsmod` 1: `rdw` and `rsw` outside the channel, bias-dependent resistances
    /// behind the drain and the source in series with `rsh`'s.
    One,
}

impl TryFrom<u8> for DrainSourceResistance {
    type Error = String;

    fn try_from(rdsmod: u8) -> Result<Self, String> {
        match rdsmod {
            0 => Ok(Self::Zero),
            1 => Ok(Self::One),
            _ => Err(format!(
                "BSIM4's drain_source_resistance_model is 0 or 1, not {rdsmod}"
            )),
        }
    }
}

impl From<DrainSourceResistance> for u8 {
    fn from(rdsmod: DrainSourceResistance) -> Self {
        match rdsmod {
            DrainSourceResistance::Zero => 0,
            DrainSourceResistance::One => 1,
        }
    }
}

/// The flicker noise model, `fnoimod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum FlickerNoise {
    /// `fnoimod` 0: SPICE2's, `kf · |Id|^af / (f^ef · Leff² · Coxe)`.
    Zero,
    /// `fnoimod` 1: BSIM4's of the oxide traps `noia`, `noib` and `noic`.
    One,
}

impl TryFrom<u8> for FlickerNoise {
    type Error = String;

    fn try_from(fnoimod: u8) -> Result<Self, String> {
        match fnoimod {
            0 => Ok(Self::Zero),
            1 => Ok(Self::One),
            _ => Err(format!(
                "BSIM4's flicker_noise_model is 0 or 1, not {fnoimod}"
            )),
        }
    }
}

impl From<FlickerNoise> for u8 {
    fn from(fnoimod: FlickerNoise) -> Self {
        match fnoimod {
            FlickerNoise::Zero => 0,
            FlickerNoise::One => 1,
        }
    }
}

/// The channel's thermal noise model, `tnoimod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum ThermalNoise {
    /// `tnoimod` 0: the charge-based model, of the inversion charge.
    Zero,
    /// `tnoimod` 1: the holistic model, of the conductances, which moves part of the
    /// channel's noise to the source side through `rnoia`, `rnoib`, `tnoia` and `tnoib`.
    One,
}

impl TryFrom<u8> for ThermalNoise {
    type Error = String;

    fn try_from(tnoimod: u8) -> Result<Self, String> {
        match tnoimod {
            0 => Ok(Self::Zero),
            1 => Ok(Self::One),
            _ => Err(format!(
                "BSIM4's thermal_noise_model is 0 or 1, not {tnoimod}"
            )),
        }
    }
}

impl From<ThermalNoise> for u8 {
    fn from(tnoimod: ThermalNoise) -> Self {
        match tnoimod {
            ThermalNoise::Zero => 0,
            ThermalNoise::One => 1,
        }
    }
}

/// A MOSFET in BSIM4.5, SPICE's level 14 and 54 model of short channels, as ngspice 47
/// implements it.
///
/// The drain current folds in the threshold's short-channel, narrow-width, pocket implant
/// and drain-induced barrier lowering terms, the poly gate's depletion, the mobility's
/// degradation by the gate field, Coulomb scattering and the channel's length, the
/// carriers' velocity saturation and overshoot, the source end's velocity limit, the
/// channel-length modulation, drain-induced barrier lowering and threshold shift, and the
/// substrate current's effect on the output resistance, and the source and drain
/// resistance `rdsw`. The impact ionization's substrate current flows from the drain to
/// the body, the gate-induced drain and source leakages from each to the body, and two
/// junctions join the body to the drain and the source, each carrying a diode's current
/// with its trap-assisted tunneling. Every parameter of a binned card has its length,
/// width and area dependences, and each one scales with the temperature as `b4v5temp.c`
/// scales it; the stress of the length of diffusion moves the threshold, the mobility and
/// the saturation velocity. The transistor is symmetric: with the drain below the source,
/// the two swap.
///
/// A sheet resistance `rsh` adds an internal node behind the drain and the source where
/// `nrd` or `nrs` is positive, labelled `drain` and `source`, and `rdsmod` 1 one behind
/// each always, through bias-dependent resistances in series with `rsh`'s; one below
/// [`SMALL_RESISTANCE`](crate::SMALL_RESISTANCE) also carries its current as an unknown,
/// labelled `drain_current` or `source_current`. With `rbodymod` 1, three internal nodes,
/// `body`, `drain_body` and `source_body`, lie under the channel and the junctions.
///
/// The charges under the gate follow `capmod` 2, the charge-thickness model, with the
/// drain's share of the channel's charge `xpart`, beside the overlaps' charges with their
/// bias dependence, and each junction holds a depletion charge. The channel makes the
/// thermal noise of `tnoimod` and the flicker noise of `fnoimod`, and the drain, source
/// and body resistances thermal noise. The gate tunneling currents, the bias-dependent
/// drain and source resistances, the gate resistance, the non-quasi-static models and
/// the well proximity effect are left out.
#[derive(Debug, Clone, Copy)]
pub struct Bsim4;

/// Where a BSIM4 transistor's contributions go among its local unknowns: through its
/// drain and source resistances, and among its gate, bulk, the drain and source behind the
/// resistances and the body's nodes under the channel and the junctions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bsim4Shape {
    stamp: Nodes<7>,
    drain_series: Series,
    source_series: Series,
    /// At `rdsmod` 1, the gate, the body, the drain and source behind their
    /// resistances and the drain and source terminals, in the order of [`EXTERNAL`].
    external: Option<Nodes<6>>,
}

/// What a BSIM4 transistor remembers.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bsim4Memory {
    /// The gate-source, drain-source and bulk-source voltages, and the drain body's and
    /// the source body's over the source, of the last evaluation, from which the next
    /// steps are limited.
    last: [f64; 5],
    /// The threshold of the last evaluation, signed as the channel conducts; before the
    /// first, or since the transistor was prepared at another temperature, its zero-bias
    /// one.
    threshold: Option<f64>,
    /// The values of [`VARIABLE_NAMES`] at the last evaluation.
    variables: [f64; 5],
}

/// The variables a BSIM4 transistor reports: its threshold and saturation voltages, in
/// its own polarity, and its region among [`REGIONS`] and its margins.
const VARIABLE_NAMES: [&str; 5] = [
    "threshold_voltage",
    "saturation_voltage",
    REGION,
    OVERDRIVE,
    SATURATION_MARGIN,
];

/// The regions of a transistor's channel, on either side of its threshold and saturation
/// voltages, in the order of [`REGIONS`]' names. The equations blend one region into the
/// next through `Vgsteff` and `Vdseff`, and switch at neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Region {
    /// Below its threshold, its current falling exponentially.
    WeakInversion,
    /// At or above its threshold, with the drain below saturation.
    Linear,
    /// At or above its threshold, with the drain at or past saturation.
    Saturated,
}

/// The regions a transistor's equations distinguish, and its margins to their boundaries:
/// `Vgst = Vgs_eff − Vth` and `Vds − Vdsat`, in an n-channel's sign, with the drain and
/// the source swapped where the drain lies below the source, as the equations swap them.
const REGIONS: Regions = Regions {
    names: &["weak_inversion", "linear", "saturated"],
    margins: &[OVERDRIVE, SATURATION_MARGIN],
};

/// Local node indices among the stamped ones: the gate and the bulk, the drain and source
/// behind their resistances, and the body's nodes under the channel, the drain junction
/// and the source junction, each the bulk itself at `rbodymod` 0.
const GATE: usize = 0;
const BULK: usize = 1;
const INNER_DRAIN: usize = 2;
const INNER_SOURCE: usize = 3;
const BODY: usize = 4;
const DRAIN_BODY: usize = 5;
const SOURCE_BODY: usize = 6;

/// The pairs of nodes across which the voltages the channel's equations take as variables
/// lie: gate-source, drain-source and body-source.
const VARIABLES: [(usize, usize); 3] = [
    (GATE, INNER_SOURCE),
    (INNER_DRAIN, INNER_SOURCE),
    (BODY, INNER_SOURCE),
];

/// Local node indices among the nodes of `rdsmod` 1's resistances.
const EXTERNAL: [usize; 6] = [0, 1, 2, 3, 4, 5];
const EXTERNAL_GATE: usize = EXTERNAL[0];
const EXTERNAL_BODY: usize = EXTERNAL[1];
const EXTERNAL_INNER_DRAIN: usize = EXTERNAL[2];
const EXTERNAL_INNER_SOURCE: usize = EXTERNAL[3];
const EXTERNAL_DRAIN: usize = EXTERNAL[4];
const EXTERNAL_SOURCE: usize = EXTERNAL[5];

/// The pairs of nodes across which the voltages `rdsmod` 1's drain and source
/// resistances take as variables lie: the gate's, the body's and the terminal's over the
/// node behind each.
const EXTERNAL_VARIABLES: [[(usize, usize); 3]; 2] = [
    [
        (EXTERNAL_GATE, EXTERNAL_INNER_DRAIN),
        (EXTERNAL_BODY, EXTERNAL_INNER_DRAIN),
        (EXTERNAL_DRAIN, EXTERNAL_INNER_DRAIN),
    ],
    [
        (EXTERNAL_GATE, EXTERNAL_INNER_SOURCE),
        (EXTERNAL_BODY, EXTERNAL_INNER_SOURCE),
        (EXTERNAL_SOURCE, EXTERNAL_INNER_SOURCE),
    ],
];

/// The pairs of nodes of the five body resistances: the channel's body to the drain's,
/// to the source's and to the bulk, and the drain's and the source's to the bulk.
const BODY_RESISTANCES: [(usize, usize); 5] = [
    (BODY, DRAIN_BODY),
    (BODY, SOURCE_BODY),
    (BODY, BULK),
    (DRAIN_BODY, BULK),
    (SOURCE_BODY, BULK),
];

/// The names of the five body resistances' thermal noise, in the order of
/// [`BODY_RESISTANCES`].
const BODY_NOISE: [&str; 5] = [
    "rbpd thermal",
    "rbps thermal",
    "rbpb thermal",
    "rbdb thermal",
    "rbsb thermal",
];

/// The five voltages a transistor limits: gate-source, drain-source and body-source, and
/// the drain body's and the source body's over the source.
const LIMITED: [(usize, usize); 5] = [
    (GATE, INNER_SOURCE),
    (INNER_DRAIN, INNER_SOURCE),
    (BODY, INNER_SOURCE),
    (DRAIN_BODY, INNER_SOURCE),
    (SOURCE_BODY, INNER_SOURCE),
];

/// What a transistor carries at one bias, oriented as an n-channel's: the channel's
/// currents with their derivatives with respect to the gate-source, drain-source and
/// body-source voltages, and the junctions' with respect to their own voltages.
struct Terminals<R: Real> {
    /// The channel's current from the drain to the source.
    channel: D<R>,
    /// The currents from the drain and from the source into the body: the substrate
    /// current from whichever acts as the drain, and each one's gate-induced leakage.
    drain_body: D<R>,
    source_body: D<R>,
    /// The currents from the drain's and the source's body nodes into the drain and the
    /// source junctions, `gmin` across each.
    junctions: [J<R>; 2],
    /// The charges on the gate, the body and the drain against the source.
    charges: [D<R>; 3],
    /// The junctions' depletion charges, on the drain's and the source's body nodes
    /// against the drain and the source.
    depletions: [J<R>; 2],
    /// The threshold and saturation voltages.
    threshold: f64,
    saturation: f64,
    region: Region,
    /// `Vgst` and `Vds − Vdsat`.
    margins: [f64; 2],
}

/// `f`, a function of the gate-source, drain-source and body-source voltages of the frame
/// whose source is the drain, with its derivatives with respect to the terminals' own
/// voltages.
fn unswapped<R: Real>(f: D<R>) -> D<R> {
    let [gate, drain, bulk] = f.gradient;
    D {
        value: f.value,
        gradient: [gate, -(gate + drain + bulk), bulk],
    }
}

/// `b4v5ld.c`'s and `b4v5temp.c`'s `DEXP`: `e^x`, continued linearly above
/// [`EXP_THRESHOLD`] and held at [`MIN_EXP`] below its negative.
fn dexp<R: Real, const N: usize>(x: Dual<R, N>) -> Dual<R, N> {
    let value = x.value.value();
    if value > EXP_THRESHOLD {
        (x - EXP_THRESHOLD + 1.0) * crate::kernel::bsim3::MAX_EXP
    } else if value < -EXP_THRESHOLD {
        Dual::number(MIN_EXP)
    } else {
        x.exp()
    }
}

impl<R: Real> Junction<R> {
    /// The current from the body into the junction at its voltage `v`, `diomod` 1's diode
    /// with its trap-assisted tunneling, `gmin` across it.
    fn current(&self, v: J<R>, gmin: f64) -> J<R> {
        let leak = v * gmin;
        let saturation = self.saturation_current;
        let slope = self.slope;
        let diode = if saturation.value() <= 0.0 {
            J::number(0.0)
        } else if (v.divide(slope)).value.value() < -EXP_THRESHOLD {
            J::constant(saturation * (MIN_EXP - 1.0))
        } else if v.value.value() <= self.vjm.value() {
            (v.divide(slope).exp() - 1.0).scale(saturation)
        } else {
            let current = self.current_at_vjm;
            v.plus(-self.vjm)
                .scale(current / slope)
                .plus(current - saturation)
        };
        let tunneling =
            self.tunneling
                .iter()
                .fold(J::number(0.0), |sum, &(current, vts, slope)| {
                    let ratio = if (vts - v.value).value() < vts.value() * 1e-3 {
                        dexp(-(v.divide(slope)) * 1.0e3)
                    } else {
                        dexp(-(v.divide(slope)).scale(vts) / (J::constant(vts) - v))
                    };
                    sum + (ratio - 1.0).scale(current)
                });
        diode - tunneling + leak
    }
}

impl<R: Real> External<R> {
    /// The conductance of one copy's drain resistance, `side` 0, or source resistance, 1,
    /// `rsh`'s in series with it, at the gate's and the body's voltages `vg` and `vb` over
    /// the node behind it, oriented as an n-channel's.
    fn conductance(&self, side: usize, [vg, vb]: [D<R>; 2]) -> D<R> {
        let pick = |values: [R; 2]| if side == 0 { values[0] } else { values[1] };
        let t0 = vg.plus(-self.vfbsd);
        let vg_eff = (t0 + (t0 * t0 + 1.0e-4).sqrt()) * 0.5;
        let t0 = vg_eff.scale(self.prwg) + 1.0;
        let t2 = t0.recip() - vb.scale(self.prwb);
        let t3 = t2 + (t2 * t2 + 0.01).sqrt();
        let resistance = t3
            .scale(pick(self.resistances) * 0.5)
            .plus(pick(self.minimums));
        let sheet = pick(self.sheet);
        (resistance.scale(sheet) + 1.0).recip().scale(sheet)
    }

    /// The conductances of the drain's and the source's resistances of every copy, at
    /// `at`.
    fn conductances(&self, nodes: &Nodes<6>, sign: f64, multiplier: R, at: &Local<'_>) -> [f64; 2] {
        let [drain, source] = EXTERNAL_VARIABLES;
        [(0, drain), (1, source)].map(|(side, variables)| {
            let [gate, body, _] = variables
                .map(|(plus, minus)| sign * (nodes.voltage(at, plus) - nodes.voltage(at, minus)));
            let [vg, vb] = [gate, body].map(D::number);
            (self.conductance(side, [vg, vb]).value * multiplier).value()
        })
    }

    /// Adds the currents through the drain's and the source's resistances of every copy,
    /// from each terminal to the node behind it, and their derivatives.
    fn add(
        &self,
        nodes: &Nodes<6>,
        sign: f64,
        multiplier: R,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let mut derivatives = Derivatives::new();
        for (side, variables) in EXTERNAL_VARIABLES.into_iter().enumerate() {
            let applied = variables
                .map(|(plus, minus)| sign * (nodes.voltage(at, plus) - nodes.voltage(at, minus)));
            let bias = parts::Bias::new(applied, applied);
            let [vg, vb, across] = Dual::variables(applied);
            let current = (self.conductance(side, [vg, vb]) * across).scale(multiplier);
            let oriented = Oriented {
                nodes,
                sign,
                bias: &bias,
                variables,
            };
            oriented.add_current(stamp, &mut derivatives, variables[2], current);
        }
        derivatives.stamp(nodes, stamp);
    }
}

impl<R: Real> PreparedBsim4<R> {
    /// What the transistor carries at the gate-source, drain-source and body-source
    /// voltages `voltages`, and the junctions at theirs, `junctions`, the drain's then
    /// the source's, with `gmin` across each junction.
    #[expect(
        clippy::similar_names,
        reason = "the voltages' and overlaps' names in BSIM4"
    )]
    fn terminals(&self, voltages: [f64; 3], junctions: [f64; 2], gmin: f64) -> Terminals<R> {
        let [gate, drain, bulk] = voltages;
        let forward = drain >= 0.0;
        let frame = if forward {
            voltages
        } else {
            [gate - drain, -drain, bulk - drain]
        };
        let core = &self.core;
        let variables = Dual::variables(frame);
        let operating = core.evaluate(variables);
        let oriented = |f: D<R>| if forward { f } else { unswapped(f) };
        let ids = oriented(operating.ids);
        let isub = oriented(operating.isub);
        let [vgs, vds, vbs] = Dual::variables(voltages);
        let intrinsic = if core.xpart < 0.0 {
            // `b4v5ld.c` gives the channel no charge with a negative `xpart`.
            [D::number(0.0); 3]
        } else {
            let charges = core.intrinsic(variables[1], &operating);
            [charges.gate, charges.bulk, charges.drain].map(oriented)
        };
        let [qg, qb, qd] = intrinsic;
        let [qgso, qgdo, qgbo] = self.overlaps.charges([vgs, vgs - vds, vgs - vbs]);
        let gate_charge = qg + qgdo + qgso + qgbo;
        let bulk_charge = qb - qgbo;
        let drain_charge = if forward {
            qd - qgdo
        } else {
            // In the swapped frame the intrinsic drain charge is the source's.
            -(gate_charge + bulk_charge + (qd - qgso))
        };
        let [gidl, gisl] = core.gate_induced_leakages([vgs, vds, vbs]);
        let (drain_body, source_body) = if forward {
            (isub + gidl, gisl)
        } else {
            (gidl, isub + gisl)
        };
        let margins = [
            (operating.vgs_eff - operating.vth).value.value(),
            (variables[1] - operating.vdsat).value.value(),
        ];
        let region = match margins {
            [overdrive, _] if overdrive < 0.0 => Region::WeakInversion,
            [_, margin] if margin < 0.0 => Region::Linear,
            _ => Region::Saturated,
        };
        let [vbd_jct, vbs_jct] = junctions.map(|v| J::variables([v])[0]);
        Terminals {
            channel: if forward { ids } else { -ids },
            drain_body,
            source_body,
            junctions: [
                self.drain_junction.current(vbd_jct, gmin),
                self.source_junction.current(vbs_jct, gmin),
            ],
            charges: [gate_charge, bulk_charge, drain_charge],
            depletions: [
                self.drain_depletion.charge(vbd_jct),
                self.source_depletion.charge(vbs_jct),
            ],
            threshold: operating.vth.value.value(),
            saturation: operating.vdsat.value.value(),
            region,
            margins,
        }
    }

    /// The five limited voltages at `at`, limited from the last evaluation's, which
    /// `memory` keeps and takes the new ones.
    fn bias(
        &self,
        shape: &Bsim4Shape,
        memory: &mut Bsim4Memory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) -> parts::Bias<5> {
        let nodes = &shape.stamp;
        let sign = self.core.sign;
        let applied = LIMITED
            .map(|(plus, minus)| sign * (nodes.voltage(at, plus) - nodes.voltage(at, minus)));
        let limited = self.limit(applied, memory).filter(|_| at.limiting);
        if limited.is_some() {
            stamp.mark_limited();
        }
        let voltages = limited.unwrap_or(applied);
        memory.last = voltages;
        parts::Bias::new(applied, voltages)
    }

    /// The `applied` voltages limited from the last evaluation's, which `memory` keeps, as
    /// `b4v5ld.c` limits them: the gate's step around the threshold and the drain's, from
    /// whichever end acts as the source, then the steps of the body and of each junction's
    /// body node, from the thermal voltage at 27 °C and the critical voltage of a
    /// saturation current of 10 fA. Returns `None` when no step needs limiting.
    #[expect(clippy::similar_names, reason = "the voltages' names in BSIM4")]
    fn limit(&self, applied: [f64; 5], memory: &Bsim4Memory) -> Option<[f64; 5]> {
        let [vgs, vds, vbs, vdbs, vsbs] = applied;
        let [
            previous_gate,
            previous_drain,
            previous_bulk,
            previous_dbs,
            previous_sbs,
        ] = memory.last;
        let threshold = memory
            .threshold
            .unwrap_or(self.core.sign * self.core.vth0.value());
        let ([vgs, vds_limited], channel) =
            limit_channel([vgs, vds], [previous_gate, previous_drain], threshold);
        // `b4v5ld.c` limits the junctions' steps from ngspice's thermal voltage at 27 °C
        // and the critical voltage of 10 fA, whatever the junctions' own.
        let critical = critical_voltage();
        let junction = |new: f64, old: f64| limit_junction(new, old, VT0, critical);
        // Each step is limited against the end that acts as the source.
        let from = |v: f64, previous: f64| {
            if vds_limited >= 0.0 {
                junction(v, previous).map_or((v, false), |limited| (limited, true))
            } else {
                junction(v - vds, previous - previous_drain)
                    .map_or((v - vds + vds_limited, false), |limited| {
                        (limited + vds_limited, true)
                    })
            }
        };
        let (vbs, body) = from(vbs, previous_bulk);
        let (vdbs, drain) = if self.body_network.is_some() {
            from(vdbs, previous_dbs)
        } else {
            (vbs, false)
        };
        let (vsbs, source) = if self.body_network.is_some() {
            from(vsbs, previous_sbs)
        } else {
            (vbs, false)
        };
        (channel || body || drain || source).then_some([vgs, vds_limited, vbs, vdbs, vsbs])
    }
}

impl Kernel for Bsim4 {
    const NAME: &'static str = "bsim4";
    type Parameters<R: Real> = Bsim4Parameters<R>;
    type Shape = Bsim4Shape;
    type Prepared<R: Real> = PreparedBsim4<R>;
    type Memory = Bsim4Memory;

    fn interface<R: Real>(_: &Bsim4Parameters<R>) -> Interface {
        Interface::new(
            &["drain", "gate", "source", "bulk"],
            &[
                Branch::new(0, 2, BranchKind::Conductive),
                Branch::new(3, 0, BranchKind::Conductive),
                Branch::new(3, 2, BranchKind::Conductive),
                Branch::new(1, 2, BranchKind::Capacitive),
                Branch::new(1, 0, BranchKind::Capacitive),
                Branch::new(1, 3, BranchKind::Capacitive),
            ],
        )
    }

    fn prepare<R: Real>(
        parameters: &Bsim4Parameters<R>,
        temperature: R,
    ) -> Result<PreparedBsim4<R>, ParameterError> {
        parameters.prepared(temperature)
    }

    fn layout<R: Real>(
        parameters: &Bsim4Parameters<R>,
        _: &PreparedBsim4<R>,
        layout: &mut Layout,
    ) -> Result<Bsim4Shape, ParameterError> {
        let [drain_resistance, source_resistance] = parameters.resistances()?;
        let [drain, gate, source, bulk] = [0, 1, 2, 3];
        let external = parameters.drain_source_resistance_model == DrainSourceResistance::One;
        let (drain_series, inner_drain) = if external {
            (Series::None, layout.internal("drain", Nature::Voltage))
        } else {
            Series::behind(layout, drain, drain_resistance, ("drain", "drain_current"))
        };
        let (source_series, inner_source) = if external {
            (Series::None, layout.internal("source", Nature::Voltage))
        } else {
            Series::behind(
                layout,
                source,
                source_resistance,
                ("source", "source_current"),
            )
        };
        let [body, drain_body, source_body] = match parameters.body_resistance_model {
            BodyResistance::Zero => [bulk; 3],
            BodyResistance::One => ["body", "drain_body", "source_body"]
                .map(|label| layout.internal(label, Nature::Voltage)),
        };
        let nodes = [
            gate,
            bulk,
            inner_drain,
            inner_source,
            body,
            drain_body,
            source_body,
        ];
        let stamp = Nodes::new(layout, nodes);
        let external = external.then(|| {
            Nodes::new(
                layout,
                [gate, body, inner_drain, inner_source, drain, source],
            )
        });
        Ok(Bsim4Shape {
            stamp,
            drain_series,
            source_series,
            external,
        })
    }

    #[expect(clippy::similar_names, reason = "the voltages' names in BSIM4")]
    fn evaluate<R: Real>(
        shape: &Bsim4Shape,
        prepared: &PreparedBsim4<R>,
        memory: &mut Bsim4Memory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [drain_series, source_series] = prepared.series;
        shape.drain_series.add(drain_series, at, stamp);
        shape.source_series.add(source_series, at, stamp);
        let nodes = &shape.stamp;
        if let Some(conductances) = prepared.body_network {
            for ((plus, minus), conductance) in BODY_RESISTANCES.into_iter().zip(conductances) {
                let voltage = nodes.voltage(at, plus) - nodes.voltage(at, minus);
                nodes.current(stamp, plus, minus, conductance * voltage);
                nodes.conductance(stamp, (plus, minus), (plus, minus), conductance);
            }
        }
        let bias = prepared.bias(shape, memory, at, stamp);
        let [vgs, vds, vbs, vdbs, vsbs] = bias.voltages;
        let [
            gate_beyond,
            drain_beyond,
            bulk_beyond,
            dbs_beyond,
            sbs_beyond,
        ] = bias.beyond;
        let channel_bias = parts::Bias::new(
            [vgs + gate_beyond, vds + drain_beyond, vbs + bulk_beyond],
            [vgs, vds, vbs],
        );
        let junction_bias =
            |voltage: f64, beyond: f64| parts::Bias::new([voltage + beyond], [voltage]);
        let drain_junction = junction_bias(vdbs - vds, dbs_beyond - drain_beyond);
        let source_junction = junction_bias(vsbs, sbs_beyond);
        let terminals = prepared.terminals(
            channel_bias.voltages,
            [drain_junction.voltages[0], source_junction.voltages[0]],
            at.gmin,
        );
        memory.threshold = Some(terminals.threshold);
        let sign = prepared.core.sign;
        let [overdrive, margin] = terminals.margins;
        memory.variables = [
            sign * terminals.threshold,
            sign * terminals.saturation,
            f64::from(terminals.region as u8),
            overdrive,
            margin,
        ];
        let channel = Oriented {
            nodes,
            sign,
            bias: &channel_bias,
            variables: VARIABLES,
        };
        let drain_side = Oriented {
            nodes,
            sign,
            bias: &drain_junction,
            variables: [(DRAIN_BODY, INNER_DRAIN)],
        };
        let source_side = Oriented {
            nodes,
            sign,
            bias: &source_junction,
            variables: [(SOURCE_BODY, INNER_SOURCE)],
        };
        let mut derivatives = Derivatives::new();
        let sums = &mut derivatives;
        // `b4v5ld.c` multiplies every current of one copy, `gmin`'s too, by `m`.
        let copies = |value: D<R>| value.scale(prepared.multiplier);
        let [ids, drain_body, source_body] = [
            terminals.channel,
            terminals.drain_body,
            terminals.source_body,
        ]
        .map(copies);
        let [drain_jct, source_jct] = terminals
            .junctions
            .map(|value| value.scale(prepared.multiplier));
        let [gate, bulk, drain] = terminals.charges.map(copies);
        let [drain_depletion, source_depletion] = terminals
            .depletions
            .map(|value| value.scale(prepared.multiplier));
        channel.add_current(stamp, sums, (INNER_DRAIN, INNER_SOURCE), ids);
        channel.add_current(stamp, sums, (INNER_DRAIN, BODY), drain_body);
        channel.add_current(stamp, sums, (INNER_SOURCE, BODY), source_body);
        drain_side.add_current(stamp, sums, (DRAIN_BODY, INNER_DRAIN), drain_jct);
        source_side.add_current(stamp, sums, (SOURCE_BODY, INNER_SOURCE), source_jct);
        channel.add_charge(stamp, sums, (GATE, INNER_SOURCE), gate);
        channel.add_charge(stamp, sums, (BODY, INNER_SOURCE), bulk);
        channel.add_charge(stamp, sums, (INNER_DRAIN, INNER_SOURCE), drain);
        drain_side.add_charge(stamp, sums, (DRAIN_BODY, INNER_DRAIN), drain_depletion);
        source_side.add_charge(stamp, sums, (SOURCE_BODY, INNER_SOURCE), source_depletion);
        derivatives.stamp(nodes, stamp);
        if let (Some(nodes), Some(external)) = (&shape.external, &prepared.external) {
            external.add(nodes, sign, prepared.multiplier, at, stamp);
        }
    }

    /// The zero-bias threshold, which a transistor's first step is limited from, moves
    /// with its temperature.
    fn reheat(memory: &mut Bsim4Memory) {
        memory.threshold = None;
    }

    fn variable_names() -> &'static [&'static str] {
        &VARIABLE_NAMES
    }

    fn variables(memory: &Bsim4Memory, values: &mut [f64]) {
        parts::write_variables(&memory.variables, values);
    }

    fn regions() -> Regions {
        REGIONS
    }

    /// The thermal noise of the drain, source and body resistances, and the channel's
    /// thermal noise of `tnoimod` and flicker noise of `fnoimod`, as `b4v5noi.c` computes
    /// them.
    fn noise<R: Real>(
        shape: &Bsim4Shape,
        prepared: &PreparedBsim4<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let temperature = prepared.temperature;
        let nodes = &shape.stamp;
        if let Some(conductances) = prepared.body_network {
            let names = BODY_RESISTANCES.iter().zip(BODY_NOISE).zip(conductances);
            for ((&(plus, minus), name), conductance) in names {
                let equations = [nodes.equation(plus), nodes.equation(minus)];
                sources.push(NoiseSource::thermal(
                    name,
                    equations,
                    conductance.value(),
                    temperature,
                ));
            }
        }
        let core = &prepared.core;
        let voltages = VARIABLES
            .map(|(plus, minus)| core.sign * (nodes.voltage(at, plus) - nodes.voltage(at, minus)));
        let channel = prepared.channel_noise(voltages);
        let [mut drain, mut source] = prepared.series;
        if let Some(added) = channel.source_side {
            // `tnoimod` 1 adds a resistance of the channel's noise to whichever acts as
            // the source.
            let [drain_resistance, source_resistance] = prepared.resistances;
            if channel.forward {
                source = shape.source_series.value_of(source_resistance + added);
            } else {
                drain = shape.drain_series.value_of(drain_resistance + added);
            }
        }
        if let (Some(external_nodes), Some(external)) = (&shape.external, &prepared.external) {
            // `rdsmod` 1's resistances, `tnoimod` 1's share on the source side added.
            let [mut drain, mut source] =
                external.conductances(external_nodes, core.sign, prepared.multiplier, at);
            if let Some(added) = channel.source_side {
                let shared = |conductance: f64| conductance / (1.0 + added * conductance);
                if channel.forward {
                    source = shared(source);
                } else {
                    drain = shared(drain);
                }
            }
            for (name, conductance, terminal, inner) in [
                ("drain thermal", drain, EXTERNAL_DRAIN, EXTERNAL_INNER_DRAIN),
                (
                    "source thermal",
                    source,
                    EXTERNAL_SOURCE,
                    EXTERNAL_INNER_SOURCE,
                ),
            ] {
                let equations = [
                    external_nodes.equation(terminal),
                    external_nodes.equation(inner),
                ];
                sources.push(NoiseSource::thermal(
                    name,
                    equations,
                    conductance,
                    temperature,
                ));
            }
        }
        shape
            .drain_series
            .noise(drain, ("drain thermal", temperature), sources);
        shape
            .source_series
            .noise(source, ("source thermal", temperature), sources);
        let equations = [nodes.equation(INNER_DRAIN), nodes.equation(INNER_SOURCE)];
        sources.push(NoiseSource::thermal(
            "channel thermal",
            equations,
            channel.thermal,
            temperature,
        ));
        if channel.flicker > 0.0 {
            sources.push(NoiseSource {
                name: "flicker",
                equations,
                white: 0.0,
                flicker: channel.flicker,
                exponent: prepared.noise.ef,
            });
        }
    }
}

/// What a channel's noise is at one bias, of all its copies: the conductance whose
/// thermal noise its own is, its flicker density at 1 Hz, whether its drain lies at or
/// above its source, and the resistance `tnoimod` 1 adds on its source side.
struct ChannelNoise {
    thermal: f64,
    flicker: f64,
    forward: bool,
    source_side: Option<f64>,
}

impl<R: Real> PreparedBsim4<R> {
    /// The channel's noise at the gate-source, drain-source and body-source voltages
    /// `voltages`, as `b4v5noi.c` computes it.
    fn channel_noise(&self, voltages: [f64; 3]) -> ChannelNoise {
        let [gate, drain, bulk] = voltages;
        let forward = drain >= 0.0;
        let frame = if forward {
            voltages
        } else {
            [gate - drain, -drain, bulk - drain]
        };
        let vds = frame[1];
        let operating = self.core.evaluate(Dual::variables(frame));
        let (thermal, source_side) = self.thermal_noise(&operating, vds);
        let flicker = self.flicker_noise(&operating, vds);
        ChannelNoise {
            thermal,
            flicker,
            forward,
            source_side,
        }
    }

    /// The conductance whose thermal noise the channel's of all its copies is, at the
    /// frame's drain-source voltage `vds` where the channel is `operating`, and the
    /// resistance `tnoimod` 1 adds on its source side.
    fn thermal_noise(&self, operating: &Operating<R>, vds: f64) -> (f64, Option<f64>) {
        let core = &self.core;
        let noise = &self.noise;
        let value = |number: D<R>| number.value.value();
        let copies = self.multiplier.value();
        let leff = core.leff.value();
        let ueff = value(operating.ueff);
        let vgsteff = value(operating.vgsteff);
        // `b4v5ld.c` keeps the current over `Vdseff` of one finger above 1 nA per volt.
        let ids_over_vds = value(operating.ids_over_vds).max(1e-9) * core.fingers.value();
        match noise.thermal_model {
            ThermalNoise::Zero => {
                let abulk = value(operating.abulk0) * core.abulk_cv_factor.value();
                let vdsat = vgsteff / abulk;
                let t0 = vdsat - vds - DELTA;
                let t1 = (t0 * t0 + 4.0 * DELTA * vdsat).sqrt();
                let vdseff = if vds == 0.0 {
                    0.0
                } else if t0 >= 0.0 {
                    vdsat - f64::midpoint(t0, t1)
                } else {
                    vdsat * (1.0 - 2.0 * DELTA / (t1 - t0))
                };
                let t0 = abulk * vdseff;
                let t1 = 12.0 * (vgsteff - 0.5 * t0 + 1.0e-20);
                let t3 = t0 * vdseff / t1;
                let qinv = value(operating.coxeff)
                    * (core.weff_cv * core.leff_cv * core.fingers).value()
                    * (vgsteff - 0.5 * t0 + abulk * t3);
                let t0 = ueff * qinv.abs();
                let t1 = t0 * value(operating.rds) + leff * leff;
                (copies * t0 / t1 * noise.ntnoi, None)
            }
            ThermalNoise::One => {
                let t5 = vgsteff / value(operating.esat_l);
                let t5 = t5 * t5;
                let beta = noise.rnoia * (1.0 + t5 * noise.tnoia * leff);
                let theta = noise.rnoib * (1.0 + t5 * noise.tnoib * leff);
                let [gm, gds, gmbs] = operating.ids.gradient.map(Real::value);
                let sum = gm + gmbs + gds;
                let induced = theta * theta * sum * sum / ids_over_vds;
                let t1 = beta * (gm + gmbs) + gds;
                let thermal = copies * (t1 * t1 / ids_over_vds - induced);
                (thermal, Some(theta * theta / (copies * ids_over_vds)))
            }
        }
    }

    /// The flicker density at 1 Hz of the channel's copies, at the frame's drain-source
    /// voltage `vds` where the channel is `operating`.
    #[expect(
        clippy::similar_names,
        reason = "b4v5noi.c's names of the noise's terms"
    )]
    fn flicker_noise(&self, operating: &Operating<R>, vds: f64) -> f64 {
        let core = &self.core;
        let noise = &self.noise;
        let value = |number: D<R>| number.value.value();
        let copies = self.multiplier.value();
        let leff = core.leff.value();
        let coxe = core.coxe.value();
        let ueff = value(operating.ueff);
        let vgsteff = value(operating.vgsteff);
        let vdseff = value(operating.vdseff);
        let current = value(operating.ids).abs();
        let fingers = core.fingers.value();
        let vtm = core.vtm.value();
        match noise.flicker_model {
            FlickerNoise::Zero => {
                copies * noise.kf * (noise.af * current.max(N_MINLOG).ln()).exp()
                    / (leff * leff * coxe)
            }
            FlickerNoise::One => {
                let temperature = self.temperature;
                let nstar = value(operating.nstar);
                let length = noise.length;
                let leffsq = length * length;
                let esat = 2.0 * core.vsattemp.value() / ueff;
                let litl = core.litl.value();
                let clm = if noise.em <= 0.0 {
                    0.0
                } else {
                    let t0 = ((vds - vdseff) / litl + noise.em) / esat;
                    (litl * t0.max(N_MINLOG).ln()).max(0.0)
                };
                let weff = core.weff.value() * fingers;
                let t1 =
                    NOISE_CHARGE * NOISE_CHARGE * NGSPICE_BOLTZMANN * current * temperature * ueff;
                let t2 = 1.0e10 * value(operating.abulk_unscaled) * coxe * leffsq;
                let abov_vgst2vtm = value(operating.abulk) / (vgsteff + 2.0 * vtm);
                let n0 = coxe * vgsteff / NOISE_CHARGE;
                let nl = coxe * vgsteff * (1.0 - abov_vgst2vtm * vdseff) / NOISE_CHARGE;
                let t3 = noise.noia * ((n0 + nstar) / (nl + nstar)).max(N_MINLOG).ln();
                let t4 = noise.noib * (n0 - nl);
                let t5 = noise.noic * 0.5 * (n0 * n0 - nl * nl);
                let t6 = NGSPICE_BOLTZMANN * temperature * current * current;
                let t7 = 1.0e10 * leffsq * weff;
                let t8 = noise.noia + noise.noib * nl + noise.noic * nl * nl;
                let t9 = (nl + nstar) * (nl + nstar);
                let strong = t1 / t2 * (t3 + t4 + t5) + t6 / t7 * clm * t8 / t9;
                let t10 = noise.noia * NGSPICE_BOLTZMANN * temperature;
                let t11 = weff * leff * 1.0e10 * nstar * nstar;
                let weak = t10 / t11 * current * current;
                let both = strong + weak;
                if both > 0.0 {
                    copies * strong * weak / both
                } else {
                    0.0
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::Differentiable;
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_mirrors,
        assert_parameter_gradients_match, assert_switches_where_margin_vanishes, source,
    };

    /// A short n-channel with every effect of the drain current and the charges switched
    /// on: the poly gate's depletion, the pocket implant, the parasitic resistance, the
    /// non-saturation effect, the velocity overshoot and the source end's limit, the
    /// substrate current, the gate-induced leakages, the junctions' linear continuation and
    /// trap-assisted tunneling, and the overlaps' bias dependence.
    const SHORT: &str = r#"{"width": 2e-6, "length": 0.15e-6, "toxe": 4.1e-9, "toxp": 3.8e-9,
        "vth0": 0.45, "u0": 0.03, "eu": 1.67, "k1": 0.5, "k2": -0.02, "ngate": 1e26,
        "dvtp0": 1e-8, "dvtp1": 0.5, "rdsw": 200, "prwg": 0.1, "prwb": -0.1, "a1": 0.1,
        "a2": 0.9, "lambda": 1e-10, "vtl": 2e5, "alpha0": 1e-6, "beta0": 20, "agidl": 1e-9,
        "pvag": 0.5, "pdits": 0.2, "pditsd": 0.3, "fprout": 0.2, "keta": -0.03,
        "etab": -0.1, "cit": 1e-4, "cdscd": 1e-3, "jtss": 1e-7, "jtssws": 1e-12,
        "drain_area": 1e-12, "source_area": 1.2e-12, "drain_perimeter": 4e-6,
        "source_perimeter": 5e-6, "cgsl": 1e-10, "cgdl": 1.2e-10, "dwg": -5e-9,
        "dwb": 3e-9}"#;

    /// The same transistor with a p-channel.
    fn p_channel(parameters: &str) -> String {
        parameters
            .replace(r#""vth0": 0.45"#, r#""vth0": -0.45"#)
            .replace('{', r#"{"channel": "p", "#)
    }

    /// `parameters` with `more`, a list of JSON members, before them.
    fn with(parameters: &str, more: &str) -> String {
        parameters.replacen('{', &format!("{{{more}, "), 1)
    }

    proptest! {
        /// The conductances and capacitances are the derivatives of the currents and
        /// charges in every region, in both modes, for both channels, of every charge
        /// partition and both body networks, with the bulk up to 0.6 V forward of the
        /// drain or the source.
        #[test]
        fn jacobians_match_finite_differences(
            drain in -2.0..2.0f64,
            gate in -1.0..2.0f64,
            reverse in -0.6..1.5f64,
            xpart in proptest::sample::select(vec![0.0, 0.5, 1.0]),
            body in 0..2u8,
            external in 0..2u8,
            p in proptest::bool::ANY,
        ) {
            let models = format!(
                r#""charge_partition": {xpart}, "body_resistance_model": {body},
                    "drain_source_resistance_model": {external}, "rdw": 150, "rsw": 120,
                    "prwg": 0.2, "prwb": -0.1, "rdwmin": 10, "rswmin": 8, "rsh": 5,
                    "drain_squares": 2, "source_squares": 1.5"#
            );
            // The substrate current's `e^(−β0/ΔV)` curves so sharply, as `(β0/ΔV²)²`, and
            // kinks where `ΔV = β0/34`, that central differences of the body's conductance
            // err by up to 4e-3 of it: `substrate_current_jacobians_match_finite_differences`
            // checks it where the drain is well past saturation.
            let parameters = with(SHORT, &models).replace(r#""alpha0": 1e-6, "#, "");
            let parameters = if p { p_channel(&parameters) } else { parameters };
            let sign = if p { -1.0 } else { 1.0 };
            let bulk = drain.min(0.0) - reverse;
            let mut x = vec![sign * drain, sign * gate, 0.0, sign * bulk];
            // The drain and the source behind their resistances, which `rsh` alone makes
            // at `rdsmod` 0 too, away from where their currents balance.
            x.extend([sign * 0.95 * drain, sign * 0.01]);
            if body == 1 {
                // The body's nodes near the bulk, away from where their currents balance.
                x.extend([0.9, 1.0, 1.1].map(|share| sign * bulk * share));
            }
            assert_jacobians_match::<Bsim4>(&parameters, &x);
        }
    }

    #[test]
    fn substrate_current_jacobians_match_finite_differences() {
        for external in [0, 1] {
            let models = format!(
                r#""drain_source_resistance_model": {external}, "rsh": 5,
                "drain_squares": 2, "source_squares": 1.5"#
            );
            let n = with(SHORT, &models);
            for (parameters, sign) in [(p_channel(&n), -1.0), (n, 1.0)] {
                for [drain, gate] in [[1.8, 1.0], [-1.8, 1.2]] {
                    let x = [drain, gate, 0.0, drain.min(0.0) - 0.3, 0.95 * drain, 0.01]
                        .map(|value| sign * value);
                    assert_jacobians_match::<Bsim4>(&parameters, &x);
                }
            }
        }
    }

    #[test]
    fn parameter_gradients_are_exact() {
        let binned = r#""lvth0": 2e-9, "wvth0": -1e-9, "pvth0": 1e-16, "lu0": 1e-10,
            "wua": 1e-17, "pk2": 1e-16, "lrdsw": 1e-6, "wvsat": 1e-3, "lcgsl": 1e-18,
            "nominal_temperature": 290, "kt1l": 1e-9, "ute": -1.3, "prt": 20, "tpb": 1e-3,
            "tcj": 1e-3, "rsh": 15, "drain_squares": 1, "source_squares": 1"#;
        let parameters = with(SHORT, binned);
        for [drain, gate, bulk] in [
            [1.5, 1.2, -0.5],
            // Strongly inverted, the subthreshold swing moves the currents by 1e-15 of
            // themselves, which central differences no longer resolve: the gate stays
            // below 1 V here.
            [0.3, 0.9, -0.2],
            [-1.0, 1.2, -1.5],
            [1.5, 0.3, -0.1],
        ] {
            // The internal nodes away from where their currents balance, which would leave
            // differences of their residuals only rounding.
            let inner = [drain, gate, 0.0, bulk, 0.8 * drain, 0.1];
            assert_parameter_gradients_match::<Bsim4>(&parameters, &inner);
        }
    }

    #[test]
    fn a_p_channel_mirrors_an_n_channel() {
        for x in [
            [1.5, 1.2, 0.0, -0.5],
            [-1.0, 1.5, 0.0, -1.5],
            [1.0, 0.3, 0.0, 0.2],
        ] {
            assert_mirrors(&Differentiable::<Bsim4>::new(), SHORT, &x);
        }
    }

    #[test]
    fn swapping_the_drain_and_the_source_reverses_the_channel() {
        let symmetric = SHORT
            .replace(r#""source_area": 1.2e-12"#, r#""source_area": 1e-12"#)
            .replace(r#""source_perimeter": 5e-6"#, r#""source_perimeter": 4e-6"#)
            .replace(r#""cgsl": 1e-10"#, r#""cgsl": 1.2e-10"#)
            .replace(r#""jtssws": 1e-12"#, r#""jtssws": 0"#)
            .replace(r#""jtss": 1e-7"#, r#""jtss": 0"#);
        let mut bench = Bench::new::<Bsim4>(&symmetric);
        let forward = bench.settle(&[1.5, 1.8, 0.0, -0.5]);
        let reversed = bench.settle(&[0.0, 1.8, 1.5, -0.5]);
        assert_close(&[forward.residual[0]], &[reversed.residual[2]]);
        assert_close(
            &[forward.charge[0], forward.charge[1]],
            &[reversed.charge[2], reversed.charge[1]],
        );
    }

    #[test]
    fn rejects_what_ngspice_stops_on() {
        let error = |parameters: &str| Bench::try_new::<Bsim4>(parameters).unwrap_err().0;
        assert_eq!(error(r#"{"toxe": 0}"#), "toxe must be positive, not 0.0");
        assert_eq!(
            error(r#"{"length": 1e-7, "lint": 6e-8}"#),
            "the effective length must be positive, not -1.9999999999999994e-8"
        );
        assert_eq!(
            error(r#"{"dvt1": -0.1}"#),
            "dvt1 must be at least 0, not -0.1"
        );
        assert_eq!(error(r#"{"pclm": 0}"#), "pclm must be positive, not 0.0");
        assert_eq!(
            error(r#"{"fingers": 0.5}"#),
            "nf must be at least 1, not 0.5"
        );
        assert_eq!(
            error(r#"{"body_resistance_model": 2}"#),
            "jolty's BSIM4 has body_resistance_model 0 or 1, not 2"
        );
    }

    #[test]
    fn unknowns_follow_the_sheet_resistance_and_the_body_network() {
        let bench = Bench::new::<Bsim4>(
            r#"{"rsh": 10, "source_squares": 0.05, "body_resistance_model": 1}"#,
        );
        assert_eq!(
            bench.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "n3",
                "device:source",
                "device:source_current",
                "device:body",
                "device:drain_body",
                "device:source_body"
            ]
        );
        // `b4v5set.c` makes no node behind a diffusion of no squares.
        assert_eq!(
            Bench::new::<Bsim4>(r#"{"rsh": 10, "drain_squares": 0}"#)
                .unknowns()
                .len(),
            4
        );
    }

    #[test]
    fn its_channel_and_resistances_make_the_noise_of_its_noise_models() {
        let x = [1.5, 1.2, 0.0, 0.0];
        // Without the substrate current and the leakages, the drain's current is the
        // channel's.
        let quiet = SHORT
            .replace(r#""alpha0": 1e-6, "#, "")
            .replace(r#""agidl": 1e-9, "#, "")
            .replace(r#""jtss": 1e-7, "#, "");
        let spice2 = with(
            &quiet,
            r#""flicker_noise_model": 0, "thermal_noise_model": 1, "kf": 1e-26, "af": 1.2,
                "ef": 0.9, "rnoia": 1, "rnoib": 0, "tnoia": 0, "tnoib": 0"#,
        );
        let mut bench = Bench::new::<Bsim4>(&spice2);
        let current = bench.settle(&x).residual[0];
        let sources = bench.noise(&x);
        let leff = 0.15e-6;
        let coxe = 3.9 * EPS0 / 4.1e-9;
        // The holistic model's share of the source side, `rnoib`², comes off the channel's
        // noise, which `rnoia` otherwise sets alone with `tnoia` and `tnoib` at 0.
        let shared = with(&spice2, r#""rnoib": 0.5"#).replace(r#", "rnoib": 0"#, "");
        let channel = |sources: &[NoiseSource]| source(sources, "channel thermal").density(1.0);
        assert_close(
            &[channel(&Bench::new::<Bsim4>(&shared).noise(&x))],
            &[0.75 * channel(&sources)],
        );
        assert_close(
            &[source(&sources, "flicker").density(10.0)],
            &[1e-26 * current.powf(1.2) / (leff * leff * coxe) / 10f64.powf(0.9)],
        );
        let bsim4 = with(
            &quiet,
            r#""body_resistance_model": 1, "rsh": 10, "drain_squares": 2"#,
        );
        let sources = Bench::new::<Bsim4>(&bsim4).noise(&[1.5, 1.2, 0.0, 0.0, 1.4, 0.0, 0.0, 0.0]);
        for name in [
            "channel thermal",
            "flicker",
            "drain thermal",
            "rbpd thermal",
            "rbps thermal",
            "rbpb thermal",
            "rbdb thermal",
            "rbsb thermal",
        ] {
            assert!(source(&sources, name).density(1.0) > 0.0, "{name}");
        }
    }

    /// A transistor is in weak inversion below its threshold, linear above it with its
    /// drain below its saturation voltage, and saturated past it, each region switching
    /// where its margin vanishes.
    #[test]
    fn reports_its_region_and_margins() {
        let mut n = Bench::new::<Bsim4>(r#"{"vth0": 0.5}"#);
        for (x, region) in [
            ([1.0, 0.1, 0.0, 0.0], "weak_inversion"),
            ([0.05, 2.0, 0.0, 0.0], "linear"),
            ([3.0, 3.0, 0.0, 0.0], "saturated"),
        ] {
            n.settle(&x);
            let (found, _) = n.region();
            assert_eq!(found, region, "{x:?}");
        }
        let switches = [
            ([3.0, 0.1, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0], OVERDRIVE),
            (
                [0.05, 3.0, 0.0, 0.0],
                [3.0, 3.0, 0.0, 0.0],
                SATURATION_MARGIN,
            ),
        ];
        for (from, to, margin) in switches {
            assert_switches_where_margin_vanishes(&mut n, (&from, &to), margin);
        }
    }
}
