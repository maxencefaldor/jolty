//! BSIM4.5, the Berkeley short-channel IGFET model of CMOS processes from 130 nm down, as
//! ngspice 47 implements it in `src/spicelib/devices/bsim4v5/`, SPICE's level 14 and 54
//! MOSFET of version 4.5.
//!
//! Equations and defaults follow the BSIM4.5.0 manual (X. Xi, M. Dunga, and others, UC
//! Berkeley, 2005) as written in ngspice `b4v5ld.c`, `b4v5temp.c`, `b4v5set.c`,
//! `b4v5geo.c`, and `b4v5check.c`, including constants. Equations evaluate once on dual
//! numbers carrying voltage derivatives, yielding exact conductances and capacitances
//! for all currents and charges.
//!
//! Supported modes match the version 4.5 cards of the `SKY130` and `GF180MCU` process
//! kits: `mobmod` 0, `capmod` 2, `diomod` 1, `tempmod` 0, `geomod` 0, `igcmod` 0,
//! `igbmod` 0, `trnqsmod` 0, `acnqsmod` 0, `wpemod` 0, and `rgeomod` 0. Selectable
//! options cover `rgatemod` 0 to 3 plus `rbodymod`, `rdsmod`, `permod`, `fnoimod`, and
//! `tnoimod` (0 or 1); any unsupported mode is rejected by name.
//!
//! BSIM4 and its submodules are derived from BSIM4.5.0, developed by the UC Berkeley
//! BSIM Research Group: Copyright 2005 Regents of the University of California. All
//! rights reserved. Usage is governed by the BSIM4 terms of use reproduced in the
//! repository's `NOTICE` file.

mod channel;
mod charges;
mod parameters;
mod prepare;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use self::channel::Operating;
pub use self::parameters::Bsim4Parameters;
use self::prepare::{External, Gate, Junction, PreparedBsim4};
use crate::kernel::bsim::{
    D, DELTA, EXP_THRESHOLD, MAX_EXP, MIN_EXP, N_MINLOG, NGSPICE_BOLTZMANN, NOISE_CHARGE, REGIONS,
    Region, VARIABLE_NAMES, VT0, clm_noise_length, critical_voltage, forward_frame, fringe_factor,
    interface, unswapped, variables,
};
use crate::kernel::dual::Dual;
use crate::kernel::junction::{limit_channel, limit_junction};
use crate::kernel::parts::{self, Derivatives, Nodes, Oriented, Series};
use crate::{
    Interface, Kernel, Layout, Local, Nature, NoiseSource, ParameterError, ParameterKind,
    ParameterMetadata, Real, Regions, Stamp,
};

/// The permittivity of vacuum, in farads per meter, as `b4v5temp.c` takes it.
const EPS0: f64 = 8.85418e-12;
/// π as `b4v5set.c` and `b4v5temp.c` write it, to ten digits.
#[expect(
    clippy::approx_constant,
    reason = "b4v5set.c's and b4v5temp.c's own π, which their numbers are computed with"
)]
const PI: f64 = 3.141_592_654;

/// A junction's voltage, with its derivative with respect to itself.
type J<R> = Dual<R, 1>;

/// Computes `b4v5set.c` fringing capacitance per unit width
/// `2 · epsrox · ε0 / π · ln(1 + 0.4 µm / toxe)` for oxide `toxe` and relative
/// permittivity `epsrox`.
fn fringe<R: Real>(epsrox: R, toxe: R) -> R {
    fringe_factor(toxe) * epsrox * (2.0 * EPS0 / PI)
}

/// Body resistance network selector (`rbodymod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum BodyResistance {
    /// Body model 0 (`rbodymod = 0`): channel and junctions share the bulk terminal.
    Zero,
    /// Body model 1 (`rbodymod = 1`): five resistances connect bulk to internal channel
    /// and junction body nodes.
    One,
}

super::numbered_enum!(
    BodyResistance,
    "jolty's BSIM4 has body_resistance_model 0 or 1",
    {
        0 => Zero,
        1 => One,
    }
);

/// Gate resistance network selector (`rgatemod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum GateResistance {
    /// Gate model 0 (`rgatemod = 0`): gate terminal connects directly to the channel gate.
    Zero,
    /// Gate model 1 (`rgatemod = 1`): fixed electrode resistance connects gate terminal to
    /// the channel gate node.
    One,
    /// Gate model 2 (`rgatemod = 2`): electrode resistance in series with bias-dependent
    /// intrinsic-input resistance forms one conductance to the channel gate.
    Two,
    /// Gate model 3 (`rgatemod = 3`): electrode resistance connects gate terminal to an
    /// intermediate node carrying overlap charges, and intrinsic-input resistance connects
    /// that node to the channel gate.
    Three,
}

super::numbered_enum!(
    GateResistance,
    "BSIM4's gate_resistance_model is 0, 1, 2 or 3",
    {
        0 => Zero,
        1 => One,
        2 => Two,
        3 => Three,
    }
);

/// Diffusion perimeter convention (`permod`) for `ps` and `pd`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum Perimeter {
    /// Perimeter model 0 (`permod = 0`): diffusion perimeter includes the gate edge.
    Zero,
    /// Perimeter model 1 (`permod = 1`): diffusion perimeter excludes the gate edge.
    One,
}

super::numbered_enum!(Perimeter, "BSIM4's perimeter_model is 0 or 1", {
    0 => Zero,
    1 => One,
});

/// Source and drain resistance model selector (`rdsmod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum DrainSourceResistance {
    /// Resistance model 0 (`rdsmod = 0`): folds `rdsw` into drain current and stamps fixed
    /// `rsh` behind drain and source when `nrd` or `nrs` is positive.
    Zero,
    /// Resistance model 1 (`rdsmod = 1`): places bias-dependent `rdw` and `rsw` in series
    /// with `rsh` outside the channel behind drain and source.
    One,
}

super::numbered_enum!(
    DrainSourceResistance,
    "BSIM4's drain_source_resistance_model is 0 or 1",
    {
        0 => Zero,
        1 => One,
    }
);

/// Flicker noise model selector (`fnoimod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum FlickerNoise {
    /// Flicker model 0 (`fnoimod = 0`): SPICE2 formula `kf · |Id|^af / (f^ef · Leff² · Coxe)`.
    Zero,
    /// Flicker model 1 (`fnoimod = 1`): BSIM4 oxide-trap model using `noia`, `noib`, and
    /// `noic`.
    One,
}

super::numbered_enum!(FlickerNoise, "BSIM4's flicker_noise_model is 0 or 1", {
    0 => Zero,
    1 => One,
});

/// Channel thermal noise model selector (`tnoimod`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum ThermalNoise {
    /// Thermal model 0 (`tnoimod = 0`): charge-based model driven by inversion charge.
    Zero,
    /// Thermal model 1 (`tnoimod = 1`): holistic conductance model shifting part of
    /// channel noise to the source side via `rnoia`, `rnoib`, `tnoia`, and `tnoib`.
    One,
}

super::numbered_enum!(ThermalNoise, "BSIM4's thermal_noise_model is 0 or 1", {
    0 => Zero,
    1 => One,
});

/// MOSFET using BSIM4.5 (SPICE levels 14 and 54) for short channels, matching ngspice 47.
///
/// Drain current includes short-channel, narrow-width, pocket-implant, and DIBL threshold
/// shifts, polysilicon gate depletion, gate-field, Coulomb, and length mobility
/// degradation, velocity saturation and overshoot, and source-end velocity limiting.
/// Output conductance includes channel-length modulation, DIBL, drain-induced threshold
/// shift, substrate-current body effect, and `rdsw`. Impact ionization drives substrate
/// current from drain to body, GIDL and GISL leak from drain and source to body, and
/// bulk-drain and bulk-source junctions carry diode currents with trap-assisted tunneling.
/// Binned parameters scale with length, width, area, temperature (`b4v5temp.c`), and
/// length-of-diffusion stress, and drain and source swap symmetrically when the drain lies
/// below the source.
///
/// Positive sheet resistance `rsh` with `nrd` or `nrs` adds internal nodes `drain` and
/// `source`, and `rdsmod = 1` always adds both behind bias-dependent resistances in series
/// with `rsh`. Fixed resistances below [`SMALL_RESISTANCE`](crate::SMALL_RESISTANCE) add
/// branch-current unknowns `drain_current` or `source_current`. Setting `rbodymod = 1`
/// adds internal body nodes `body`, `drain_body`, and `source_body`. Setting `rgatemod` 1
/// to 3 adds internal node `gate` behind electrode resistance `rshg` (combined in series
/// with intrinsic-input resistance at mode 2). Mode 3 splits electrode and intrinsic-input
/// resistances at `middle_gate`, which carries gate overlap charges.
///
/// Intrinsic gate charges follow `capmod = 2` (charge-thickness model) with `xpart`
/// partitioning, alongside bias-dependent overlap charges and junction depletion charges.
/// Noise covers `tnoimod` channel thermal noise, `fnoimod` flicker noise, and thermal
/// noise for drain, source, body, and gate-electrode resistances. Gate tunneling currents,
/// non-quasi-static models, and well proximity effects are omitted.
#[derive(Debug, Clone, Copy)]
pub struct Bsim4;

/// Local matrix layout for [`Bsim4`] covering drain/source series resistances, nine
/// intrinsic and resistance-network nodes, and optional `rdsmod = 1` external nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bsim4Shape {
    stamp: Nodes<9>,
    drain_series: Series,
    source_series: Series,
    /// At `rdsmod` 1, the gate, the body, the drain and source behind their
    /// resistances and the drain and source terminals, in the order of [`EXTERNAL`].
    external: Option<Nodes<6>>,
}

/// Step-limiting history and cached operating variables for [`Bsim4`].
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

/// Local indices among the nine stamped nodes: channel gate, bulk, internal drain,
/// internal source, body nodes, middle gate, and gate terminal. Channel, drain-junction,
/// and source-junction body nodes alias `BULK` at `rbodymod = 0`. Both `MIDDLE_GATE`
/// (except at `rgatemod = 3`) and `GATE` (at `rgatemod = 0`) alias `INNER_GATE`.
const INNER_GATE: usize = 0;
const BULK: usize = 1;
const INNER_DRAIN: usize = 2;
const INNER_SOURCE: usize = 3;
const BODY: usize = 4;
const DRAIN_BODY: usize = 5;
const SOURCE_BODY: usize = 6;
const MIDDLE_GATE: usize = 7;
const GATE: usize = 8;

/// The pairs of nodes across which the voltages the channel's equations take as variables
/// lie: gate-source, drain-source and body-source.
const VARIABLES: [(usize, usize); 3] = [
    (INNER_GATE, INNER_SOURCE),
    (INNER_DRAIN, INNER_SOURCE),
    (BODY, INNER_SOURCE),
];

/// The pairs of nodes across which the voltages the overlaps' charges take as variables
/// lie: the gate's middle over the source, drain-source and body-source.
const MIDDLE_VARIABLES: [(usize, usize); 3] = [
    (MIDDLE_GATE, INNER_SOURCE),
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
    (INNER_GATE, INNER_SOURCE),
    (INNER_DRAIN, INNER_SOURCE),
    (BODY, INNER_SOURCE),
    (DRAIN_BODY, INNER_SOURCE),
    (SOURCE_BODY, INNER_SOURCE),
];

/// Evaluated n-channel-oriented currents, charges, gate conductance, and operating
/// variables at one bias, carrying derivatives with respect to channel and junction
/// voltages.
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
    /// The channel's charges on the gate, the body and the drain against the source.
    charges: [D<R>; 3],
    /// The overlaps' charges on the gate's middle, the body and the drain against the
    /// source, with their derivatives with respect to the voltages of
    /// [`MIDDLE_VARIABLES`].
    overlaps: [D<R>; 3],
    /// The conductance of one copy from the gate terminal, at `rgatemod` 2, or from the
    /// gate's middle, at 3, to the channel's gate, which depends on the bias.
    gate: Option<D<R>>,
    /// The junctions' depletion charges, on the drain's and the source's body nodes
    /// against the drain and the source.
    depletions: [J<R>; 2],
    /// The threshold and saturation voltages.
    threshold: f64,
    saturation: f64,
    region: Region,
    /// Boundary margins `Vgst` and `Vds - Vdsat`.
    margins: [f64; 2],
}

impl<R: Real> Terminals<R> {
    /// Packs [`VARIABLE_NAMES`] values for a channel with polarity `sign`.
    fn variables(&self, sign: f64) -> [f64; 5] {
        variables(
            sign,
            self.threshold,
            self.saturation,
            self.region,
            self.margins,
        )
    }
}

/// Adds the currents through the five body resistances of `rbodymod` 1, of
/// `conductances` in the order of [`BODY_RESISTANCES`], and their derivatives.
fn add_body_network<R: Real>(
    nodes: &Nodes<9>,
    conductances: [R; 5],
    at: &Local<'_>,
    stamp: &mut Stamp<'_, R>,
) {
    for ((plus, minus), conductance) in BODY_RESISTANCES.into_iter().zip(conductances) {
        let voltage = nodes.unknown::<R>(at, plus) - nodes.unknown::<R>(at, minus);
        nodes.current(stamp, plus, minus, conductance * voltage);
        nodes.conductance(stamp, (plus, minus), (plus, minus), conductance);
    }
}

/// Evaluates `b4v5ld.c` and `b4v5temp.c` `DEXP`: `e^x` continued linearly above
/// [`EXP_THRESHOLD`] and floored at [`MIN_EXP`] below `-EXP_THRESHOLD`.
fn dexp<R: Real, const N: usize>(x: Dual<R, N>) -> Dual<R, N> {
    let value = x.value.value();
    if value > EXP_THRESHOLD {
        (x - EXP_THRESHOLD + 1.0) * MAX_EXP
    } else if value < -EXP_THRESHOLD {
        Dual::number(MIN_EXP)
    } else {
        x.exp()
    }
}

impl<R: Real> Junction<R> {
    /// Computes body-to-diffusion diode current at voltage `v` under `diomod = 1` with
    /// trap-assisted tunneling and shunt `gmin`.
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
    /// Computes single-copy drain (`side = 0`) or source (`side = 1`) external conductance
    /// in series with `rsh` at n-channel-oriented `[vg, vb]`.
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

    /// Computes total external drain and source conductances across all copies at `at`.
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
            let oriented = Oriented {
                nodes,
                sign,
                bias: &bias,
                variables,
            };
            let [vg, vb, across] = Dual::variables(oriented.across(at));
            let current = (self.conductance(side, [vg, vb]) * across).scale(multiplier);
            oriented.add_current(stamp, &mut derivatives, variables[2], current);
        }
        derivatives.stamp(nodes, stamp);
    }
}

impl<R: Real> Gate<R> {
    /// Computes single-copy bias-dependent gate conductance into the channel gate for
    /// `operating`. Mode 2 places `b4v5ld.c` intrinsic-input conductance `gcrg` in series
    /// with the electrode, mode 3 returns `gcrg`, and modes 0 and 1 return `None`.
    fn conductance(&self, operating: &Operating<R>) -> Option<D<R>> {
        let intrinsic =
            (operating.beta.scale(self.xrcrg2_vtm) + operating.ids_over_vds).scale(self.xrcrg1_nf);
        match self.model {
            GateResistance::Zero | GateResistance::One => None,
            GateResistance::Two => {
                Some(intrinsic.scale(self.electrode) / intrinsic.plus(self.electrode))
            }
            GateResistance::Three => Some(intrinsic),
        }
    }
}

impl<R: Real> Gate<R> {
    /// Adds gate-resistance currents and derivatives for `multiplier` copies with polarity
    /// `sign`. Stamps the fixed electrode resistance at `rgatemod` 1 and 3, plus any
    /// bias-dependent `conductance` from `GATE` or `MIDDLE_GATE` to `INNER_GATE`.
    fn add(
        &self,
        nodes: &Nodes<9>,
        (sign, multiplier): (f64, R),
        (bias, conductance): (&parts::Bias<3>, Option<D<R>>),
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
        derivatives: &mut Derivatives<R, 9>,
    ) {
        if matches!(self.model, GateResistance::One | GateResistance::Three) {
            let electrode = self.electrode * multiplier;
            let voltage = nodes.unknown::<R>(at, GATE) - nodes.unknown::<R>(at, MIDDLE_GATE);
            nodes.current(stamp, GATE, MIDDLE_GATE, electrode * voltage);
            nodes.conductance(stamp, (GATE, MIDDLE_GATE), (GATE, MIDDLE_GATE), electrode);
        }
        let Some(conductance) = conductance else {
            return;
        };
        let from = if self.model == GateResistance::Two {
            GATE
        } else {
            MIDDLE_GATE
        };
        // Gate branch current equals the bias-dependent conductance times the applied drop.
        let across = sign * (nodes.voltage(at, from) - nodes.voltage(at, INNER_GATE));
        let [gate, drain, bulk] = bias.voltages;
        let [gate_beyond, drain_beyond, bulk_beyond] = bias.beyond;
        let input = parts::Bias {
            voltages: [gate, drain, bulk, across],
            beyond: [gate_beyond, drain_beyond, bulk_beyond, 0.0],
        };
        let across = Dual::variable(nodes.across::<R>(at, (from, INNER_GATE), (sign, across)));
        let current = conductance.embed([0, 1, 2]) * across.embed([3]);
        let [gate_source, drain_source, body_source] = VARIABLES;
        let oriented = Oriented {
            nodes,
            sign,
            bias: &input,
            variables: [gate_source, drain_source, body_source, (from, INNER_GATE)],
        };
        oriented.add_current(stamp, derivatives, (from, INNER_GATE), current);
    }
}

impl<R: Real> PreparedBsim4<R> {
    /// Evaluates channel currents, overlap and intrinsic charges, gate conductance, and
    /// junction responses at channel `[vgs, vds, vbs]`, `middle` offset, and `junctions`
    /// with shunt `gmin`.
    #[expect(
        clippy::similar_names,
        reason = "the voltages' and overlaps' names in BSIM4"
    )]
    fn terminals(&self, voltages: [R; 3], middle: R, junctions: [R; 2], gmin: f64) -> Terminals<R> {
        let [gate, drain, bulk] = voltages;
        let (forward, frame) = forward_frame(voltages);
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
        // In the swapped frame the intrinsic drain charge is the source's.
        let qd = if forward { qd } else { -(qg + qb + qd) };
        // `b4v5ld.c` puts the overlaps' charges on the gate's middle, at its voltages
        // over the source, the drain and the body.
        let [vgms, vds_middle, vbs_middle] = Dual::variables([gate + middle, drain, bulk]);
        let [qgso, qgdo, qgbo] =
            self.overlaps
                .charges([vgms, vgms - vds_middle, vgms - vbs_middle]);
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
        let region = Region::classify(margins);
        let [vbd_jct, vbs_jct] = junctions.map(|v| J::variables([v])[0]);
        Terminals {
            channel: if forward { ids } else { -ids },
            drain_body,
            source_body,
            junctions: [
                self.drain_junction.current(vbd_jct, gmin),
                self.source_junction.current(vbs_jct, gmin),
            ],
            charges: [qg, qb, qd],
            overlaps: [qgso + qgdo + qgbo, -qgbo, -qgdo],
            gate: self
                .gate
                .and_then(|gate| gate.conductance(&operating))
                .map(oriented),
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

    /// Reads the five limited voltages at `at`, applies step limiting against `memory`,
    /// and stores the resulting bias.
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

    /// Limits `applied` voltages against `memory`, matching `b4v5ld.c`. Gate and drain
    /// steps limit around threshold from the active source end, and body and junction
    /// nodes limit using 27 °C thermal voltage and a 10 fA critical voltage. Returns
    /// `None` when no step requires limiting.
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

    fn parameter_kinds() -> &'static [(&'static str, ParameterKind)] {
        Bsim4Parameters::<f64>::PARAMETER_KINDS
    }

    fn binned_parameters() -> &'static [&'static str] {
        Bsim4Parameters::<f64>::BINNED_PARAMETERS
    }

    fn interface<R: Real>(_: &Bsim4Parameters<R>) -> Interface {
        interface()
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
        let gate_model = parameters.gate_resistance_model;
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
        let inner_gate = if gate_model == GateResistance::Zero {
            gate
        } else {
            layout.internal("gate", Nature::Voltage)
        };
        let middle_gate = if gate_model == GateResistance::Three {
            layout.internal("middle_gate", Nature::Voltage)
        } else {
            inner_gate
        };
        let [body, drain_body, source_body] = match parameters.body_resistance_model {
            BodyResistance::Zero => [bulk; 3],
            BodyResistance::One => ["body", "drain_body", "source_body"]
                .map(|label| layout.internal(label, Nature::Voltage)),
        };
        let nodes = [
            inner_gate,
            bulk,
            inner_drain,
            inner_source,
            body,
            drain_body,
            source_body,
            middle_gate,
            gate,
        ];
        let stamp = Nodes::new(layout, nodes);
        let external = external.then(|| {
            Nodes::new(
                layout,
                [inner_gate, body, inner_drain, inner_source, drain, source],
            )
        });
        Ok(Bsim4Shape {
            stamp,
            drain_series,
            source_series,
            external,
        })
    }

    #[expect(
        clippy::similar_names,
        clippy::too_many_lines,
        reason = "the voltages' names in BSIM4, and each contribution b4v5ld.c loads, in turn"
    )]
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
            add_body_network(nodes, conductances, at, stamp);
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
        let sign = prepared.core.sign;
        let middle = sign * (nodes.voltage(at, MIDDLE_GATE) - nodes.voltage(at, INNER_GATE));
        let channel = Oriented {
            nodes,
            sign,
            bias: &channel_bias,
            variables: VARIABLES,
        };
        // The overlaps' voltages move from those evaluated at as the channel's do.
        let middle_bias = parts::Bias {
            voltages: [vgs + middle, vds, vbs],
            beyond: channel_bias.beyond,
        };
        let overlapping = Oriented {
            nodes,
            sign,
            bias: &middle_bias,
            variables: MIDDLE_VARIABLES,
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
        let [drain_junction, source_junction] =
            [&drain_side, &source_side].map(|side| side.across(at)[0]);
        let terminals = prepared.terminals(
            channel.across(at),
            nodes.across(at, (MIDDLE_GATE, INNER_GATE), (sign, middle)),
            [drain_junction, source_junction],
            at.gmin,
        );
        memory.threshold = Some(terminals.threshold);
        memory.variables = terminals.variables(sign);
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
        let [overlap_gate, overlap_bulk, overlap_drain] = terminals.overlaps.map(copies);
        let [drain_depletion, source_depletion] = terminals
            .depletions
            .map(|value| value.scale(prepared.multiplier));
        channel.add_current(stamp, sums, (INNER_DRAIN, INNER_SOURCE), ids);
        channel.add_current(stamp, sums, (INNER_DRAIN, BODY), drain_body);
        channel.add_current(stamp, sums, (INNER_SOURCE, BODY), source_body);
        drain_side.add_current(stamp, sums, (DRAIN_BODY, INNER_DRAIN), drain_jct);
        source_side.add_current(stamp, sums, (SOURCE_BODY, INNER_SOURCE), source_jct);
        channel.add_charge(stamp, sums, (INNER_GATE, INNER_SOURCE), gate);
        channel.add_charge(stamp, sums, (BODY, INNER_SOURCE), bulk);
        channel.add_charge(stamp, sums, (INNER_DRAIN, INNER_SOURCE), drain);
        overlapping.add_charge(stamp, sums, (MIDDLE_GATE, INNER_SOURCE), overlap_gate);
        overlapping.add_charge(stamp, sums, (BODY, INNER_SOURCE), overlap_bulk);
        overlapping.add_charge(stamp, sums, (INNER_DRAIN, INNER_SOURCE), overlap_drain);
        drain_side.add_charge(stamp, sums, (DRAIN_BODY, INNER_DRAIN), drain_depletion);
        source_side.add_charge(stamp, sums, (SOURCE_BODY, INNER_SOURCE), source_depletion);
        if let Some(gate) = &prepared.gate {
            let conductance = terminals.gate.map(copies);
            gate.add(
                nodes,
                (sign, prepared.multiplier),
                (&channel_bias, conductance),
                at,
                stamp,
                sums,
            );
        }
        derivatives.stamp(nodes, stamp);
        if let (Some(nodes), Some(external)) = (&shape.external, &prepared.external) {
            external.add(nodes, sign, prepared.multiplier, at, stamp);
        }
    }

    /// Clears the cached threshold voltage so the next step limits against the reheated
    /// zero-bias threshold.
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

    /// Emits thermal noise for drain, source, gate-electrode, and body resistances plus
    /// channel thermal (`tnoimod`) and flicker (`fnoimod`) noise per `b4v5noi.c`.
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
        if let Some(gate) = &prepared.gate {
            // `b4v5noi.c` takes the electrode's thermal noise alone, at `rgatemod` 2 too.
            sources.push(NoiseSource::thermal(
                "gate thermal",
                [nodes.equation(GATE), nodes.equation(MIDDLE_GATE)],
                (gate.electrode * prepared.multiplier).value(),
                temperature,
            ));
        }
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

/// Evaluated channel noise across all parallel copies at one bias: equivalent thermal
/// conductance, 1 Hz flicker density, forward-orientation flag, and optional source-side
/// resistance from `tnoimod = 1`.
struct ChannelNoise {
    thermal: f64,
    flicker: f64,
    forward: bool,
    source_side: Option<f64>,
}

impl<R: Real> PreparedBsim4<R> {
    /// Computes channel thermal and flicker noise at `[vgs, vds, vbs]`, matching
    /// `b4v5noi.c`.
    fn channel_noise(&self, voltages: [f64; 3]) -> ChannelNoise {
        let (forward, frame) = forward_frame(voltages);
        let vds = frame[1];
        let operating = self.core.evaluate(Dual::variables(frame.map(R::constant)));
        let (thermal, source_side) = self.thermal_noise(&operating, vds);
        let flicker = self.flicker_noise(&operating, vds);
        ChannelNoise {
            thermal,
            flicker,
            forward,
            source_side,
        }
    }

    /// Computes total equivalent thermal-noise conductance and optional `tnoimod = 1`
    /// source-side resistance for `operating` at forward-frame `vds`.
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

    /// Computes total 1 Hz channel flicker-noise density across all parallel copies at
    /// forward-frame `vds` and `operating`.
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
                let clm = clm_noise_length(vds, vdseff, litl, esat, noise.em);
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
    use crate::Thermal;
    use crate::kernel::region::{OVERDRIVE, SATURATION_MARGIN};
    use crate::testing::{
        Bench, assert_close, assert_jacobians_match, assert_model_jacobians_match,
        assert_switches_where_margin_vanishes, assert_unknown_derivatives_match, source,
    };

    /// Short n-channel fixture exercising all active drain-current and charge effects:
    /// polysilicon gate depletion, pocket implant, parasitic resistance, non-saturation,
    /// velocity overshoot, and source-end velocity limiting. The fixture includes
    /// substrate current, GIDL/GISL, diode linear continuation, trap-assisted tunneling,
    /// and bias-dependent overlap charges.
    const SHORT: &str = r#"{"width": 2e-6, "length": 0.15e-6, "toxe": 4.1e-9, "toxp": 3.8e-9,
        "vth0": 0.45, "u0": 0.03, "eu": 1.67, "k1": 0.5, "k2": -0.02, "ngate": 1e26,
        "dvtp0": 1e-8, "dvtp1": 0.5, "rdsw": 200, "prwg": 0.1, "prwb": -0.1, "a1": 0.1,
        "a2": 0.9, "lambda": 1e-10, "vtl": 2e5, "alpha0": 1e-6, "beta0": 20, "agidl": 1e-9,
        "pvag": 0.5, "pdits": 0.2, "pditsd": 0.3, "fprout": 0.2, "keta": -0.03,
        "etab": -0.1, "cit": 1e-4, "cdscd": 1e-3, "jtss": 1e-7, "jtssws": 1e-12,
        "drain_area": 1e-12, "source_area": 1.2e-12, "drain_perimeter": 4e-6,
        "source_perimeter": 5e-6, "cgsl": 1e-10, "cgdl": 1.2e-10, "dwg": -5e-9,
        "dwb": 3e-9}"#;

    /// Returns `parameters` rewritten for a p-channel device.
    fn p_channel(parameters: &str) -> String {
        parameters
            .replace(r#""vth0": 0.45"#, r#""vth0": -0.45"#)
            .replace('{', r#"{"channel": "p", "#)
    }

    /// Returns `parameters` with JSON fields `more` prepended.
    fn with(parameters: &str, more: &str) -> String {
        parameters.replacen('{', &format!("{{{more}, "), 1)
    }

    proptest! {
        /// Verifies analytical conductances and capacitances against finite differences
        /// across all operating regions, modes, channels, charge partitions, body
        /// networks, and gate networks, including forward bulk bias up to 0.6 V.
        #[test]
        fn jacobians_match_finite_differences(
            drain in -2.0..2.0f64,
            gate in -1.0..2.0f64,
            reverse in -0.6..1.5f64,
            xpart in proptest::sample::select(vec![0.0, 0.5, 1.0]),
            body in 0..2u8,
            external in 0..2u8,
            resistive in 0..4u8,
            p in proptest::bool::ANY,
        ) {
            let models = format!(
                r#""charge_partition": {xpart}, "body_resistance_model": {body},
                    "drain_source_resistance_model": {external}, "rdw": 150, "rsw": 120,
                    "prwg": 0.2, "prwb": -0.1, "rdwmin": 10, "rswmin": 8, "rsh": 5,
                    "drain_squares": 2, "source_squares": 1.5,
                    "gate_resistance_model": {resistive}, "rshg": 20, "xgw": 1e-7,
                    "ngcon": 2, "xrcrg1": 10, "xrcrg2": 2"#
            );
            // The substrate current's `e^(-β0/ΔV)` curves so sharply, as `(β0/ΔV²)²`, and
            // kinks where `ΔV = β0/34`, that central differences of the body's conductance
            // err by up to 4e-3 of it: `substrate_current_jacobians_match_finite_differences`
            // checks it where the drain is well past saturation.
            let parameters = with(SHORT, &models).replace(r#""alpha0": 1e-6, "#, "");
            let parameters = if p { p_channel(&parameters) } else { parameters };
            let sign = if p { -1.0 } else { 1.0 };
            let bulk = drain.min(0.0) - reverse;
            let mut x = vec![sign * drain, sign * gate, 0.0, sign * bulk];
            // The drain and the source behind their resistances, which `rsh` alone makes
            // at `rdsmod` 0 too, and the gate behind its resistances and its middle, away
            // from where their currents balance.
            x.extend([sign * 0.95 * drain, sign * 0.01]);
            if resistive > 0 {
                x.push(sign * (0.9 * gate - 0.05));
            }
            if resistive == 3 {
                x.push(sign * (0.95 * gate - 0.02));
            }
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
        assert_eq!(error(r#"{"toxe": 0}"#), "toxe is positive, not 0.0");
        assert_eq!(
            error(r#"{"length": 1e-7, "lint": 6e-8}"#),
            "the effective length is positive, not -1.9999999999999994e-8"
        );
        assert_eq!(error(r#"{"dvt1": -0.1}"#), "dvt1 is at least 0, not -0.1");
        assert_eq!(error(r#"{"pclm": 0}"#), "pclm is positive, not 0.0");
        assert_eq!(error(r#"{"fingers": 0.5}"#), "nf is at least 1, not 0.5");
        assert_eq!(
            error(r#"{"body_resistance_model": 2}"#),
            "jolty's BSIM4 has body_resistance_model 0 or 1, not 2"
        );
        assert_eq!(
            error(r#"{"length": 1e-7, "xgl": 1e-7}"#),
            "xgl, 1e-7, is less than the drawn length plus xl, 1e-7"
        );
        assert_eq!(error(r#"{"ngcon": 0.5}"#), "ngcon is at least 1, not 0.5");
        assert_eq!(error(r#"{"njs": 0}"#), "njs is positive, not 0.0");
        assert_eq!(error(r#"{"njd": -1}"#), "njd is positive, not -1.0");
    }

    #[test]
    fn the_gate_electrodes_resistance_is_its_sheets_between_its_contacts() {
        // Gate resistance formula:
        // `rshg · (xgw + Weffcj / (3 · ngcon)) / (ngcon · nf · (l + xl - xgl))`.
        // For a 1 µm finger: `20 Ω · (0.1 µm + 1 µm / 6) / (2 · 2 · 0.14 µm)`.
        let resistance = 20.0 * (1e-7 + 1e-6 / 6.0) / (2.0 * 2.0 * 0.14e-6);
        for (model, unknowns) in [(1, 5), (3, 6)] {
            let mut bench = Bench::new::<Bsim4>(&format!(
                r#"{{"gate_resistance_model": {model}, "width": 2e-6, "length": 0.15e-6,
                    "fingers": 2, "multiplier": 3, "rshg": 20, "xgw": 1e-7, "ngcon": 2,
                    "xgl": 1e-8}}"#
            ));
            assert_eq!(bench.unknowns().len(), unknowns);
            let evaluation = bench.evaluate(&[1.0, 1.0, 0.0, 0.0, 0.9, 0.95]);
            assert_close(&[evaluation.conductance[1][1]], &[3.0 / resistance]);
        }
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
        assert_eq!(
            Bench::new::<Bsim4>(r#"{"gate_resistance_model": 3}"#).unknowns(),
            ["n0", "n1", "n2", "n3", "device:gate", "device:middle_gate"]
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
            r#""body_resistance_model": 1, "rsh": 10, "drain_squares": 2,
                "gate_resistance_model": 3"#,
        );
        let x = [1.5, 1.2, 0.0, 0.0, 1.4, 1.2, 1.2, 0.0, 0.0, 0.0];
        let sources = Bench::new::<Bsim4>(&bsim4).noise(&x);
        for name in [
            "channel thermal",
            "flicker",
            "drain thermal",
            "gate thermal",
            "rbpd thermal",
            "rbps thermal",
            "rbpb thermal",
            "rbdb thermal",
            "rbsb thermal",
        ] {
            assert!(source(&sources, name).density(1.0) > 0.0, "{name}");
        }
    }

    /// Verifies weak-inversion, linear, and saturated region classification and boundary
    /// margin zero-crossings.
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

    /// Verifies that clamping `vdseff_cv` only in value at `vds == 0` preserves its
    /// nonzero `vds` derivative `vdsat_cv / (vdsat_cv + DELTA)` and keeps terminal drain
    /// capacitances equal to `(q(h) - q(0)) / h`.
    #[test]
    fn capacitances_at_zero_drain_voltage_keep_drain_derivative() {
        let mut bench = Bench::new::<Bsim4>(SHORT);
        let at_zero = [0.0, 1.2, 0.0, -0.3];
        let step = 1e-7;
        let shifted = [step, 1.2, 0.0, -0.3];
        let base = bench.settle(&at_zero);
        let forward = bench.settle(&shifted);
        let largest = base
            .capacitance
            .iter()
            .map(|row| row[0].abs())
            .fold(0.0_f64, f64::max);
        assert!(largest > 1e-17, "expected nonzero drain capacitances");
        for row in 0..4 {
            let slope = (forward.charge[row] - base.charge[row]) / step;
            let analytic = base.capacitance[row][0];
            assert!(
                (analytic - slope).abs() <= 1e-4 * largest,
                "row {row} at vds=0: analytic {analytic:e} against right difference {slope:e}"
            );
        }
    }

    #[test]
    fn saturation_voltage_and_gate_resistance_stay_smooth_as_lambda_nears_one() {
        // At `rdsmod = 1` (channel `rds = 0`) and high reverse overdrive (`vgs' ~ 3.37 V`),
        // `1 - lambda` drops near `5e-5`. Un-rationalized `(t1 - t3) / t0` loses ~15 bits,
        // quantizing `vdsat` into `~6e-12 V` steps and breaking `rgatemod` 2 and 3 central
        // differences by `> 1e-5`.
        let drain = -1.973_028_885_834_237_5;
        let gate = 1.720_767_144_942_445_3;
        let reverse = -0.010_324_050_866_162_261;
        let bulk = drain - reverse;
        for resistive in [2u8, 3] {
            let models = format!(
                r#""charge_partition": 0.0, "body_resistance_model": 0,
                    "drain_source_resistance_model": 1, "rdw": 150, "rsw": 120,
                    "prwg": 0.2, "prwb": -0.1, "rdwmin": 10, "rswmin": 8, "rsh": 5,
                    "drain_squares": 2, "source_squares": 1.5,
                    "gate_resistance_model": {resistive}, "rshg": 20, "xgw": 1e-7,
                    "ngcon": 2, "xrcrg1": 10, "xrcrg2": 2"#
            );
            let parameters = with(SHORT, &models).replace(r#""alpha0": 1e-6, "#, "");
            let mut x = vec![
                drain,
                gate,
                0.0,
                bulk,
                0.95 * drain,
                0.01,
                0.9 * gate - 0.05,
            ];
            if resistive == 3 {
                x.push(0.95 * gate - 0.02);
            }
            assert_jacobians_match::<Bsim4>(&parameters, &x);

            // Check `vdsat` directly across tiny `1e-8 V` source steps for smooth slopes
            // (with `(t1 - t3) / t0`, a single ULP jump in `t3` makes adjacent slopes swing
            // between `1.9e-4` and `1.1e-3`).
            let mut bench = Bench::new::<Bsim4>(&parameters);
            let step = 1e-8;
            let mut values = Vec::new();
            for offset in -2..=2 {
                let mut shifted = x.clone();
                shifted[5] = 0.01 + f64::from(offset) * step;
                bench.settle(&shifted);
                values.push(bench.variable("saturation_voltage"));
            }
            let slopes: Vec<f64> = values
                .windows(2)
                .map(|pair| (pair[1] - pair[0]) / step)
                .collect();
            let spread = slopes.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - slopes.iter().copied().fold(f64::INFINITY, f64::min);
            assert!(
                spread < 1e-5,
                "rgatemod {resistive}: saturation_voltage slopes jittered by {spread:e}: {slopes:?}"
            );
        }
    }

    /// Verifies that forward-biased junction saturation current scaling preserves `d/dT`
    /// at `rise = 0.0` (`T == Tnom`).
    #[test]
    fn forward_junction_thermal_derivative_is_exact_at_nominal_temperature() {
        let model = Thermal::<Bsim4>::new();
        let smooth = SHORT.replace(r#""alpha0": 1e-6, "#, "");
        for (parameters, sign) in [(smooth.clone(), 1.0), (p_channel(&smooth), -1.0)] {
            let heated = parameters.replace('{', r#"{"thermal": true, "#);
            let [d, g, b] = [0.05, -0.2, 0.45].map(|v| sign * v);
            assert_model_jacobians_match(&model, &heated, &[d, g, 0.0, b, 0.0]);
        }
    }

    /// Verifies that second derivatives in the unknowns match finite differences across
    /// BSIM4 external drain/source, gate, and body resistance networks.
    #[test]
    fn unknown_derivatives_match_across_resistance_models() {
        let model = Thermal::<Bsim4>::new();
        for (gate_model, body_model, external_model, x) in [
            (
                2,
                1,
                1,
                vec![1.2, 1.0, 0.0, -0.4, 1.08, 0.06, 0.92, -0.36, -0.38, -0.42],
            ),
            (3, 0, 0, vec![1.2, 1.0, 0.0, -0.4, 1.05, 0.04, 0.88, 0.94]),
        ] {
            let extra = format!(
                r#""gate_resistance_model": {gate_model}, "body_resistance_model": {body_model},
                    "drain_source_resistance_model": {external_model}, "rdw": 150, "rsw": 120,
                    "rdwmin": 10, "rswmin": 8, "rsh": 5, "drain_squares": 2,
                    "source_squares": 1.5, "rshg": 20, "xgw": 1e-7, "ngcon": 2,
                    "xrcrg1": 10, "xrcrg2": 2"#
            );
            let parameters = with(SHORT, &extra).replace(r#""alpha0": 1e-6, "#, "");
            assert_unknown_derivatives_match(&model, &parameters, &x);
        }
    }
}
