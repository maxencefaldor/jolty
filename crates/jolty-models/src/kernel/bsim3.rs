//! BSIM3v3.3, the Berkeley short-channel IGFET model of most CMOS processes from 0.8 µm to
//! 0.13 µm, as ngspice 47 implements it in `src/spicelib/devices/bsim3/`, SPICE's level 8
//! and 49 MOSFET.
//!
//! The equations and their defaults are those of the BSIM3v3.3 manual (W. Liu, X. Xi and
//! others, UC Berkeley, 2005) as ngspice's `b3ld.c`, `b3temp.c`, `b3set.c` and `b3noi.c`
//! write them, constants and all. They are written once, on numbers carrying their
//! derivatives with respect to the voltages, so the conductances and capacitances are the
//! exact derivatives of the currents and charges computed.
//!
//! This model, here and in its submodules, is derived from BSIM3v3.3.0, developed by the UC
//! Berkeley BSIM Research Group: Copyright 2004 Regents of the University of California.
//! All rights reserved. It is used under the BSIM terms of use, which the repository's
//! `NOTICE` reproduces.

mod channel;
mod charges;
mod parameters;
mod prepare;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub(crate) use self::channel::{
    above_half, above_tenth, effective_drain, factor_above_tenth, non_saturation_excess,
    reciprocal_above_tenth, saturation_voltage, smooth_floor, smooth_positive,
};
pub(crate) use self::charges::{depletion, flat_band};
pub use self::parameters::Bsim3Parameters;
use self::prepare::PreparedBsim3;
use self::prepare::{Junction, Junctions};
pub(crate) use self::prepare::{check, lost};
use crate::kernel::dual::Dual;
use crate::kernel::jfet::Channel;
use crate::kernel::junction::{limit_channel, limit_junction};
use crate::kernel::parts::{self, Derivatives, Nodes, Oriented, Series};
use crate::kernel::region::{OVERDRIVE, SATURATION_MARGIN};
use crate::{
    Branch, BranchKind, Interface, Kernel, Layout, Local, NoiseSource, ParameterError, REGION,
    Real, Regions, Stamp,
};

/// Boltzmann's constant over the elementary charge, in volts per kelvin, as `b3temp.c`
/// and `b4v5temp.c` take it.
pub(crate) const KBOQ: f64 = 8.617_087e-5;
/// The permittivity of silicon dioxide, in farads per meter, as BSIM3 takes it.
const EPSOX: f64 = 3.453_133e-11;
/// The permittivity of silicon, in farads per meter, as BSIM3 and BSIM4 take it.
pub(crate) const EPSSI: f64 = 1.035_94e-10;
/// The elementary charge, in coulombs, as `b3temp.c`, `b3ld.c`, `b4v5temp.c` and
/// `b4v5ld.c` take it.
pub(crate) const Q: f64 = 1.602_19e-19;
/// `e^34`, where `b3ld.c` and `b4v5ld.c` stop their exponentials.
pub(crate) const MAX_EXP: f64 = 5.834_617_425e14;
/// `e^−34`.
pub(crate) const MIN_EXP: f64 = 1.713_908_431e-15;
/// The argument beyond which `b3ld.c` and `b4v5ld.c` stop their exponentials.
pub(crate) const EXP_THRESHOLD: f64 = 34.0;
/// The smoothing of `b3ld.c`'s and `b4v5ld.c`'s effective voltages of the charges, their
/// `DELTA_1` to `DELTA_4`, in volts.
pub(crate) const DELTA: f64 = 0.02;
/// The elementary charge, in coulombs, as ngspice's `const.h` gives it to `b3noi.c` and
/// to `b4v5ld.c`'s poly gate depletion.
pub(crate) const NOISE_CHARGE: f64 = 1.602_176_620_8e-19;
/// Boltzmann's constant over the elementary charge, in volts per kelvin, as `b3noi.c`
/// writes it.
const NOISE_KBOQ: f64 = 8.62e-5;
/// The least argument of `b3noi.c`'s logarithms, its `N_MINLOG`.
pub(crate) const N_MINLOG: f64 = 1e-38;
/// The thermal voltage at 27 °C with ngspice's constants, `CONSTvt0`, from which
/// `b3ld.c` and `b4v5ld.c` limit the bulk junctions' steps.
pub(crate) const VT0: f64 = NGSPICE_BOLTZMANN * 300.15 / NOISE_CHARGE;
/// Boltzmann's constant, in joules per kelvin, as ngspice's `const.h` gives it,
/// `CONSTboltz`, to `b4v5noi.c` and to its thermal voltage.
pub(crate) const NGSPICE_BOLTZMANN: f64 = 1.380_648_52e-23;

/// The critical voltage of a junction of 10 fA at [`VT0`], `VT0 · ln(VT0 / (√2 · 10 fA))`,
/// from which `b3ld.c` and `b4v5ld.c` limit the bulk junctions' steps, their `vcrit`.
pub(crate) fn critical_voltage() -> f64 {
    VT0 * (VT0 / (std::f64::consts::SQRT_2 * 1.0e-14)).ln()
}

/// A number and its derivatives with respect to a transistor's gate-source, drain-source
/// and bulk-source voltages.
type D<R> = Dual<R, 3>;

/// `n` for an n-channel and `p` for a p-channel.
pub(crate) const fn by_channel(channel: Channel, n: f64, p: f64) -> f64 {
    match channel {
        Channel::N => n,
        Channel::P => p,
    }
}

/// `b3set.c`'s fringing capacitance per width of an oxide `tox` thick,
/// `2 · εox / π · ln(1 + 0.4 µm / tox)`.
fn fringe<R: Real>(tox: R) -> R {
    (R::constant(0.4e-6) / tox + 1.0).ln() * (2.0 * EPSOX / std::f64::consts::PI)
}

/// The mobility model, `mobmod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum Mobility {
    /// `mobmod` 1: the gate field degrades the mobility through `Vgsteff + 2·Vth`.
    One,
    /// `mobmod` 2: through `Vgsteff` alone.
    Two,
    /// `mobmod` 3: as `mobmod` 1, the body effect a factor of its own.
    Three,
}

impl Mobility {
    /// `others` at `mobmod` 1 and 2, `three` at `mobmod` 3.
    const fn select(self, others: f64, three: f64) -> f64 {
        match self {
            Self::One | Self::Two => others,
            Self::Three => three,
        }
    }
}

impl TryFrom<u8> for Mobility {
    type Error = String;

    fn try_from(mobmod: u8) -> Result<Self, String> {
        match mobmod {
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            _ => Err(format!("BSIM3's mobility_model is 1, 2 or 3, not {mobmod}")),
        }
    }
}

impl From<Mobility> for u8 {
    fn from(mobmod: Mobility) -> Self {
        match mobmod {
            Mobility::One => 1,
            Mobility::Two => 2,
            Mobility::Three => 3,
        }
    }
}

/// The model of the charges under the gate, `capmod`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum Charges {
    /// `capmod` 0: the original long-channel charges, piecewise across the regions.
    Zero,
    /// `capmod` 1: charges continuous across the threshold but not across saturation.
    One,
    /// `capmod` 2: smooth charges.
    Two,
    /// `capmod` 3: smooth charges of an inversion layer of finite thickness.
    Three,
}

impl TryFrom<u8> for Charges {
    type Error = String;

    fn try_from(capmod: u8) -> Result<Self, String> {
        match capmod {
            0 => Ok(Self::Zero),
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            _ => Err(format!("BSIM3's capacitance_model is 0 to 3, not {capmod}")),
        }
    }
}

impl From<Charges> for u8 {
    fn from(capmod: Charges) -> Self {
        match capmod {
            Charges::Zero => 0,
            Charges::One => 1,
            Charges::Two => 2,
            Charges::Three => 3,
        }
    }
}

/// The noise model, `noimod`: SPICE2's or BSIM3's channel thermal noise and flicker
/// noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(try_from = "u8", into = "u8")]
pub enum NoiseModel {
    /// `noimod` 1: SPICE2's thermal and flicker noise.
    One,
    /// `noimod` 2: BSIM3's thermal and flicker noise.
    Two,
    /// `noimod` 3: SPICE2's thermal noise and BSIM3's flicker noise.
    Three,
    /// `noimod` 4: BSIM3's thermal noise and SPICE2's flicker noise.
    Four,
    /// `noimod` 5: SPICE2's thermal noise with ngspice's correction below saturation,
    /// and SPICE2's flicker noise.
    Five,
    /// `noimod` 6: SPICE2's thermal noise with ngspice's correction below saturation,
    /// and BSIM3's flicker noise.
    Six,
}

impl TryFrom<u8> for NoiseModel {
    type Error = String;

    fn try_from(noimod: u8) -> Result<Self, String> {
        match noimod {
            1 => Ok(Self::One),
            2 => Ok(Self::Two),
            3 => Ok(Self::Three),
            4 => Ok(Self::Four),
            5 => Ok(Self::Five),
            6 => Ok(Self::Six),
            _ => Err(format!("BSIM3's noise_model is 1 to 6, not {noimod}")),
        }
    }
}

impl From<NoiseModel> for u8 {
    fn from(noimod: NoiseModel) -> Self {
        match noimod {
            NoiseModel::One => 1,
            NoiseModel::Two => 2,
            NoiseModel::Three => 3,
            NoiseModel::Four => 4,
            NoiseModel::Five => 5,
            NoiseModel::Six => 6,
        }
    }
}

/// A MOSFET in BSIM3v3.3, SPICE's level 8 and 49 model of short channels, as ngspice 47
/// implements it.
///
/// The drain current folds in the threshold's short-channel, narrow-width and
/// drain-induced barrier lowering terms, the poly gate's depletion, the mobility's
/// degradation by the gate field in `mobmod` 1 to 3, the carriers' velocity saturation,
/// the channel-length modulation, the drain-induced barrier lowering and the substrate
/// current's effect on the output resistance, and the source and drain resistance
/// `rdsw`; the impact ionization's substrate current flows from the drain to the bulk.
/// The charges under the gate follow `capmod` 0 to 3, the drain's share of the channel's
/// charge `xpart`, beside the overlaps' charges, and two junctions join the bulk to the
/// drain and the source, each carrying a diode's current and holding a depletion
/// charge. Every parameter of a binned card has its length, width and area dependences,
/// and each one scales with the temperature as `b3temp.c` scales it. The transistor is
/// symmetric: with the drain below the source, the two swap.
///
/// A sheet resistance `rsh` adds an internal node behind the drain and the source,
/// labelled `drain` and `source`; one below [`SMALL_RESISTANCE`](crate::SMALL_RESISTANCE)
/// also carries its current as an unknown, labelled `drain_current` or `source_current`.
/// The non-quasi-static model, `nqsmod`, and the area calculation method of HSPICE,
/// `acm`, are left out.
#[derive(Debug, Clone, Copy)]
pub struct Bsim3;

/// Where a BSIM3 transistor's contributions go among its local unknowns: through its
/// drain and source resistances, and among its gate, bulk and the drain and source behind
/// the resistances.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bsim3Shape {
    stamp: Nodes<4>,
    drain_series: Series,
    source_series: Series,
}

/// What a BSIM3 transistor remembers.
#[derive(Debug, Clone, Copy, Default)]
pub struct Bsim3Memory {
    /// The gate-source, drain-source and bulk-source voltages of the last evaluation, from
    /// which the next steps are limited.
    last: [f64; 3],
    /// The threshold of the last evaluation, signed as the channel conducts; before the
    /// first, or since the transistor was prepared at another temperature, its zero-bias
    /// one.
    threshold: Option<f64>,
    /// The values of [`VARIABLE_NAMES`] at the last evaluation.
    variables: [f64; 5],
}

/// The variables a BSIM3 transistor reports: its threshold and saturation voltages, in
/// its own polarity, and its region among [`REGIONS`] and its margins.
const VARIABLE_NAMES: [&str; 5] = [
    "threshold_voltage",
    "saturation_voltage",
    REGION,
    OVERDRIVE,
    SATURATION_MARGIN,
];

/// The regions of a transistor's channel, on either side of its threshold and saturation
/// voltages, `Vth` and `Vdsat`, which bound the subthreshold, linear and saturation
/// regions the BSIM3v3.3 manual derives its drain current in, in the order of
/// [`REGIONS`]' names. The equations blend one region into the next through `Vgsteff` and
/// `Vdseff`, smooth functions that turn from one region's asymptote to the next's around
/// those voltages, and switch at neither.
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
/// `Vgst = Vgs_eff − Vth` and `Vds − Vdsat`, in an n-channel's sign as
/// [`Regions::margins`] are, with the drain and the source swapped where the drain lies
/// below the source, as the equations swap them.
const REGIONS: Regions = Regions {
    names: &["weak_inversion", "linear", "saturated"],
    margins: &[OVERDRIVE, SATURATION_MARGIN],
};

/// Local node indices: the gate and the bulk, then the drain and source behind their
/// resistances, which are the terminals themselves without one.
const GATE: usize = 0;
const BULK: usize = 1;
const INNER_DRAIN: usize = 2;
const INNER_SOURCE: usize = 3;

/// The pairs of nodes across which the voltages the transistor's equations take as
/// variables lie: gate-source, drain-source and bulk-source.
const VARIABLES: [(usize, usize); 3] = [
    (GATE, INNER_SOURCE),
    (INNER_DRAIN, INNER_SOURCE),
    (BULK, INNER_SOURCE),
];

/// The gate-source, drain-source and bulk-source voltages of one evaluation, oriented as an
/// n-channel's.
type Bias = parts::Bias<3>;

/// What a transistor carries and holds at one bias, oriented as an n-channel's, each
/// with its derivatives with respect to the gate-source, drain-source and bulk-source
/// voltages.
struct Terminals<R: Real> {
    /// The channel's current from the drain to the source.
    channel: D<R>,
    /// The substrate current and the node it flows from to the bulk, the drain or the
    /// source, whichever lies higher.
    substrate: (usize, D<R>),
    /// The currents from the bulk into the drain and the source junctions, `gmin` across
    /// each.
    junctions: [D<R>; 2],
    /// The charges on the gate, the bulk and the drain against the source.
    charges: [D<R>; 3],
    /// The junctions' depletion charges, on the bulk against the drain and the source.
    depletions: [D<R>; 2],
    /// The threshold and saturation voltages.
    threshold: f64,
    saturation: f64,
    region: Region,
    /// `Vgst` and `Vds − Vdsat`.
    margins: [f64; 2],
}

/// `f`, a function of the gate-source, drain-source and bulk-source voltages of the frame
/// whose source is the drain, with its derivatives with respect to the terminals' own
/// voltages.
fn unswapped<R: Real>(f: D<R>) -> D<R> {
    let [gate, drain, bulk] = f.gradient;
    D {
        value: f.value,
        gradient: [gate, -(gate + drain + bulk), bulk],
    }
}

impl<R: Real> Junctions<R> {
    /// The current from the bulk into `junction` at its voltage `v`, with `gmin` across it.
    fn current(&self, junction: &Junction<R>, v: D<R>, gmin: f64) -> D<R> {
        let leak = v * gmin;
        let saturation = junction.saturation_current;
        if saturation.value() <= 0.0 {
            return leak;
        }
        let slope = self.slope;
        match junction.linear {
            Some((vjm, current)) if v.value.value() >= vjm.value() => {
                v.plus(-vjm)
                    .scale(current / slope)
                    .plus(current - saturation)
                    + leak
            }
            _ => (v.divide(slope).exp() - 1.0).scale(saturation) + leak,
        }
    }
}

impl<R: Real> PreparedBsim3<R> {
    /// What the transistor carries and holds at the gate-source, drain-source and
    /// bulk-source voltages `voltages`, with `gmin` across each junction.
    #[expect(
        clippy::similar_names,
        reason = "the voltages' and overlaps' names in BSIM3"
    )]
    fn terminals(&self, voltages: [R; 3], gmin: f64) -> Terminals<R> {
        let [vgs, vds, vbs] = Dual::variables(voltages);
        let [gate, drain, bulk] = voltages;
        let forward = drain.value() >= 0.0;
        let frame = if forward {
            voltages
        } else {
            [gate - drain, -drain, bulk - drain]
        };
        let [mode_gs, mode_ds, mode_bs] = Dual::variables(frame);
        let core = &self.core;
        let operating = core.evaluate([mode_gs, mode_ds, mode_bs]);
        let intrinsic = core.intrinsic(mode_ds, mode_bs, &operating);
        let oriented = |f: D<R>| if forward { f } else { unswapped(f) };
        let ids = oriented(operating.ids);
        let isub = oriented(operating.isub);
        let [qg, qb, qd] = [intrinsic.gate, intrinsic.bulk, intrinsic.drain].map(oriented);
        let (vgd, vbd) = (vgs - vds, vbs - vds);
        let [qgso, qgdo, qgbo] = self.overlaps.charges(core.capmod, [vgs, vgd, vgs - vbs]);
        let gate_charge = qg + qgdo + qgso + qgbo;
        let bulk_charge = qb - qgbo;
        let drain_charge = if forward {
            qd - qgdo
        } else {
            // In the swapped frame the intrinsic drain charge is the source's.
            let source = qd - qgso;
            -(gate_charge + bulk_charge + source)
        };
        let junctions = &self.junctions;
        let margins = [
            (operating.vgs_eff - operating.vth).value.value(),
            (mode_ds - operating.vdsat).value.value(),
        ];
        let region = match margins {
            [overdrive, _] if overdrive < 0.0 => Region::WeakInversion,
            [_, margin] if margin < 0.0 => Region::Linear,
            _ => Region::Saturated,
        };
        Terminals {
            channel: if forward { ids } else { -ids },
            substrate: (if forward { INNER_DRAIN } else { INNER_SOURCE }, isub),
            junctions: [
                junctions.current(&junctions.drain, vbd, gmin),
                junctions.current(&junctions.source, vbs, gmin),
            ],
            charges: [gate_charge, bulk_charge, drain_charge],
            depletions: [
                junctions.charge(&junctions.drain, vbd),
                junctions.charge(&junctions.source, vbs),
            ],
            threshold: operating.vth.value.value(),
            saturation: operating.vdsat.value.value(),
            region,
            margins,
        }
    }

    /// The terminal voltages at `at`, limited from the last evaluation's, which `memory`
    /// keeps and takes the new ones.
    fn bias(
        &self,
        shape: &Bsim3Shape,
        memory: &mut Bsim3Memory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) -> Bias {
        let nodes = &shape.stamp;
        let sign = self.core.sign;
        let applied = VARIABLES
            .map(|(plus, minus)| sign * (nodes.voltage(at, plus) - nodes.voltage(at, minus)));
        let limited = self.limit(applied, memory).filter(|_| at.limiting);
        if limited.is_some() {
            stamp.mark_limited();
        }
        let voltages = limited.unwrap_or(applied);
        memory.last = voltages;
        Bias::new(applied, voltages)
    }

    /// The `applied` gate-source, drain-source and bulk-source voltages limited from the
    /// last evaluation's, which `memory` keeps, as `b3ld.c` limits them: the gate's step
    /// around the threshold and the drain's, from whichever end acts as the source, then
    /// the step of the junction with it, from the thermal voltage at 27 °C and the
    /// critical voltage of a saturation current of 10 fA. Returns `None` when no step
    /// needs limiting.
    fn limit(&self, applied: [f64; 3], memory: &Bsim3Memory) -> Option<[f64; 3]> {
        let [vgs, vds, applied_vbs] = applied;
        let [previous_gate, previous_drain, previous_bulk] = memory.last;
        let threshold = memory
            .threshold
            .unwrap_or(self.core.sign * self.core.vth0.value());
        let ([vgs, vds_limited], channel) =
            limit_channel([vgs, vds], [previous_gate, previous_drain], threshold);
        // `b3ld.c` limits the junction's step from ngspice's thermal voltage at 27 °C and
        // the critical voltage of 10 fA, whatever the junctions' own.
        let critical = critical_voltage();
        let (vbs, junction) = if vds_limited >= 0.0 {
            let limit = limit_junction(applied_vbs, previous_bulk, VT0, critical);
            (limit.unwrap_or(applied_vbs), limit.is_some())
        } else {
            let (vbd, previous_junction) = (applied_vbs - vds, previous_bulk - previous_drain);
            let limit = limit_junction(vbd, previous_junction, VT0, critical);
            (limit.unwrap_or(vbd) + vds_limited, limit.is_some())
        };
        (channel || junction).then_some([vgs, vds_limited, vbs])
    }
}

impl Kernel for Bsim3 {
    const NAME: &'static str = "bsim3";
    type Parameters<R: Real> = Bsim3Parameters<R>;
    type Shape = Bsim3Shape;
    type Prepared<R: Real> = PreparedBsim3<R>;
    type Memory = Bsim3Memory;

    fn interface<R: Real>(_: &Bsim3Parameters<R>) -> Interface {
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
        parameters: &Bsim3Parameters<R>,
        temperature: R,
    ) -> Result<PreparedBsim3<R>, ParameterError> {
        parameters.prepared(temperature)
    }

    fn layout<R: Real>(
        parameters: &Bsim3Parameters<R>,
        _: &PreparedBsim3<R>,
        layout: &mut Layout,
    ) -> Result<Bsim3Shape, ParameterError> {
        let [drain_resistance, source_resistance] = parameters.resistances()?;
        let [drain, gate, source, bulk] = [0, 1, 2, 3];
        let (drain_series, inner_drain) =
            Series::behind(layout, drain, drain_resistance, ("drain", "drain_current"));
        let (source_series, inner_source) = Series::behind(
            layout,
            source,
            source_resistance,
            ("source", "source_current"),
        );
        Ok(Bsim3Shape {
            stamp: Nodes::new(layout, [gate, bulk, inner_drain, inner_source]),
            drain_series,
            source_series,
        })
    }

    fn evaluate<R: Real>(
        shape: &Bsim3Shape,
        prepared: &PreparedBsim3<R>,
        memory: &mut Bsim3Memory,
        at: &Local<'_>,
        stamp: &mut Stamp<'_, R>,
    ) {
        let [drain_series, source_series] = prepared.series;
        shape.drain_series.add(drain_series, at, stamp);
        shape.source_series.add(source_series, at, stamp);
        let bias = prepared.bias(shape, memory, at, stamp);
        let sign = prepared.core.sign;
        let voltages = bias.across(&shape.stamp, at, (sign, VARIABLES));
        let terminals = prepared.terminals(voltages, at.gmin);
        memory.threshold = Some(terminals.threshold);
        let [overdrive, margin] = terminals.margins;
        memory.variables = [
            sign * terminals.threshold,
            sign * terminals.saturation,
            f64::from(terminals.region as u8),
            overdrive,
            margin,
        ];
        let added = Oriented {
            nodes: &shape.stamp,
            sign,
            bias: &bias,
            variables: VARIABLES,
        };
        // Every derivative among the stamped nodes is summed here, and written once. Each
        // term is added on a line of its own: added in a loop, its sums left the registers.
        let mut derivatives = Derivatives::new();
        let sums = &mut derivatives;
        // `b3ld.c` multiplies every current and charge of one copy, `gmin`'s too, by `m`.
        let copies = |value: D<R>| value.scale(prepared.multiplier);
        let (from, substrate) = terminals.substrate;
        let [channel, substrate] = [terminals.channel, substrate].map(copies);
        let [drain_junction, source_junction] = terminals.junctions.map(copies);
        let [gate, bulk, drain] = terminals.charges.map(copies);
        let [drain_depletion, source_depletion] = terminals.depletions.map(copies);
        added.add_current(stamp, sums, (INNER_DRAIN, INNER_SOURCE), channel);
        added.add_current(stamp, sums, (from, BULK), substrate);
        added.add_current(stamp, sums, (BULK, INNER_DRAIN), drain_junction);
        added.add_current(stamp, sums, (BULK, INNER_SOURCE), source_junction);
        added.add_charge(stamp, sums, (GATE, INNER_SOURCE), gate);
        added.add_charge(stamp, sums, (BULK, INNER_SOURCE), bulk);
        added.add_charge(stamp, sums, (INNER_DRAIN, INNER_SOURCE), drain);
        added.add_charge(stamp, sums, (BULK, INNER_DRAIN), drain_depletion);
        added.add_charge(stamp, sums, (BULK, INNER_SOURCE), source_depletion);
        derivatives.stamp(&shape.stamp, stamp);
    }

    /// The zero-bias threshold, which a transistor's first step is limited from, moves
    /// with its temperature.
    fn reheat(memory: &mut Bsim3Memory) {
        memory.threshold = None;
    }

    fn variable_names() -> &'static [&'static str] {
        &VARIABLE_NAMES
    }

    fn variables(memory: &Bsim3Memory, values: &mut [f64]) {
        parts::write_variables(&memory.variables, values);
    }

    fn regions() -> Regions {
        REGIONS
    }

    /// The thermal noise of the series resistances, and the channel's thermal and flicker
    /// noise of `noimod`, as `b3noi.c` computes them.
    fn noise<R: Real>(
        shape: &Bsim3Shape,
        prepared: &PreparedBsim3<R>,
        at: &Local<'_>,
        sources: &mut Vec<NoiseSource>,
    ) {
        let temperature = prepared.temperature;
        let [drain_series, source_series] = prepared.series;
        shape
            .drain_series
            .noise(drain_series, ("drain thermal", temperature), sources);
        shape
            .source_series
            .noise(source_series, ("source thermal", temperature), sources);
        let nodes = &shape.stamp;
        let core = &prepared.core;
        let voltages = VARIABLES
            .map(|(plus, minus)| core.sign * (nodes.voltage(at, plus) - nodes.voltage(at, minus)));
        let channel = [nodes.equation(INNER_DRAIN), nodes.equation(INNER_SOURCE)];
        prepared.channel_noise(voltages, channel, sources);
    }
}

impl<R: Real> PreparedBsim3<R> {
    /// Appends the channel's thermal and flicker noise at the gate-source, drain-source
    /// and bulk-source voltages `voltages`, between the `channel`'s equations.
    #[expect(clippy::similar_names, reason = "the voltages' names in BSIM3")]
    fn channel_noise(
        &self,
        voltages: [f64; 3],
        channel: [crate::Node; 2],
        sources: &mut Vec<NoiseSource>,
    ) {
        let [gate, drain, bulk] = voltages;
        let frame = if drain >= 0.0 {
            voltages
        } else {
            [gate - drain, -drain, bulk - drain]
        };
        let core = &self.core;
        let variables = Dual::variables(frame.map(R::constant));
        let operating = core.evaluate(variables);
        let [_, mode_ds, mode_bs] = variables;
        let copies = self.multiplier.value();
        let temperature = self.temperature;
        let noise = &self.noise;
        let [gm, gds, gmbs] = operating.ids.gradient.map(Real::value);
        let sum = (gm + gds + gmbs).abs();
        let leff = core.leff.value();
        let thermal = match noise.model {
            NoiseModel::One | NoiseModel::Three => 2.0 * sum / 3.0 * copies,
            NoiseModel::Five | NoiseModel::Six => {
                let vdsat = operating.vdsat.value.value();
                (3.0 - drain.min(vdsat) / vdsat) * sum / 3.0 * copies
            }
            NoiseModel::Two | NoiseModel::Four => {
                let inversion = core.intrinsic(mode_ds, mode_bs, &operating).inversion.abs();
                let ueff = operating.ueff.value.value();
                let rds = operating.rds.value.value();
                copies * ueff * inversion / (leff * leff + ueff * inversion * rds)
            }
        };
        sources.push(NoiseSource::thermal(
            "channel thermal",
            channel,
            thermal,
            temperature,
        ));
        let current = operating.ids.value.value();
        let flicker = match noise.model {
            NoiseModel::One | NoiseModel::Four | NoiseModel::Five => {
                copies * noise.kf * (noise.af * current.abs().max(N_MINLOG).ln()).exp()
                    / (leff * leff * core.cox.value())
            }
            NoiseModel::Two | NoiseModel::Three | NoiseModel::Six => {
                let strong = self.strong_inversion(&operating, drain.abs());
                let weak = noise.noia * NOISE_KBOQ * temperature
                    / (core.weff.value() * leff * 4.0e36)
                    * current
                    * current;
                let both = strong + weak;
                if both > 0.0 {
                    copies * strong * weak / both
                } else {
                    0.0
                }
            }
        };
        if flicker > 0.0 {
            sources.push(NoiseSource {
                name: "flicker",
                equations: channel,
                white: 0.0,
                flicker,
                exponent: noise.ef,
            });
        }
    }

    /// The flicker noise of BSIM3's strong inversion at 1 Hz, `b3noi.c`'s `Ssi`, where
    /// the channel is `operating` and the drain-source voltage `vds` in magnitude.
    fn strong_inversion(&self, operating: &channel::Operating<R>, vds: f64) -> f64 {
        let core = &self.core;
        let noise = &self.noise;
        let value = |number: D<R>| number.value.value();
        let current = value(operating.ids).abs();
        let leff = noise.length;
        let leffsq = leff * leff;
        let ueff = value(operating.ueff);
        let esat = 2.0 * core.vsattemp.value() / ueff;
        let litl = core.litl.value();
        let vdseff = value(operating.vdseff);
        let clm = if noise.em <= 0.0 {
            0.0
        } else {
            let t0 = ((vds - vdseff) / litl + noise.em) / esat;
            (litl * t0.max(N_MINLOG).ln()).max(0.0)
        };
        let cox = core.cox.value();
        let vgsteff = value(operating.vgsteff);
        let temperature = self.temperature;
        let t1 = NOISE_CHARGE * NOISE_CHARGE * NOISE_KBOQ * current * temperature * ueff;
        let t2 = 1.0e8 * value(operating.abulk_unscaled) * cox * leffsq;
        let abulk_over = value(operating.abulk) / (vgsteff + 2.0 * core.vtm.value());
        let n0 = cox * vgsteff / NOISE_CHARGE;
        let nl = cox * vgsteff * (1.0 - abulk_over * vdseff) / NOISE_CHARGE;
        let t3 = noise.noia * ((n0 + 2.0e14) / (nl + 2.0e14)).max(N_MINLOG).ln();
        let t4 = noise.noib * (n0 - nl);
        let t5 = noise.noic * 0.5 * (n0 * n0 - nl * nl);
        let t6 = NOISE_KBOQ * temperature * current * current;
        let t7 = 1.0e8 * leffsq * core.weff.value();
        let t8 = noise.noia + noise.noib * nl + noise.noic * nl * nl;
        let t9 = (nl + 2.0e14) * (nl + 2.0e14);
        t1 / t2 * (t3 + t4 + t5) + t6 / t7 * clm * t8 / t9
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::testing::{
        Bench, Evaluation, assert_close, assert_jacobians_match, assert_mirrors,
        assert_parameter_gradients_match, assert_switches_where_margin_vanishes, source, thermal,
    };
    use crate::{Differentiable, Input, REFERENCE_TEMPERATURE};

    /// A short n-channel with every effect of the drain current switched on: the poly
    /// gate's depletion, the parasitic resistance, the non-saturation effect, the
    /// substrate current and the diodes' linear continuation.
    const SHORT: &str = r#"{"width": 10e-6, "length": 0.35e-6, "tox": 7.6e-9,
        "nch": 1.7e23, "vth0": 0.5, "u0": 0.04, "k1": 0.6, "k2": -0.02, "rdsw": 600, "pclm": 1.2,
        "alpha0": 1e-6, "beta0": 20, "ngate": 3e26, "noff": 1.2, "voffcv": 0.05,
        "drain_area": 1e-11, "source_area": 1.2e-11, "drain_perimeter": 22e-6,
        "source_perimeter": 8e-6, "cgsl": 1e-10, "cgdl": 1.2e-10, "dwg": -5e-9,
        "dwb": 3e-9, "prwg": 0.1, "prwb": -0.1, "a1": 0.1, "a2": 0.9, "pvag": 0.5,
        "ags": 0.2, "keta": -0.03, "etab": -0.1, "cit": 1e-4, "cdscd": 1e-3,
        "ijth": 1e-4, "jsw": 1e-10}"#;

    /// The same transistor with a p-channel.
    fn p_channel(parameters: &str) -> String {
        parameters
            .replace(r#""vth0": 0.5"#, r#""vth0": -0.5"#)
            .replace('{', r#"{"channel": "p", "#)
    }

    /// `parameters` with `more`, a list of JSON members, before them.
    fn with(parameters: &str, more: &str) -> String {
        parameters.replacen('{', &format!("{{{more}, "), 1)
    }

    /// The drain current of a device with `parameters` at the drain, gate and bulk
    /// voltages `voltages`, its source grounded.
    fn drain_current(parameters: &str, [drain, gate, bulk]: [f64; 3]) -> f64 {
        Bench::new::<Bsim3>(parameters)
            .settle(&[drain, gate, 0.0, bulk])
            .residual[0]
    }

    proptest! {
        /// The conductances and capacitances are the derivatives of the currents and
        /// charges in every region, in both modes, for both channels, of every charge
        /// model and partition and every mobility model, with the bulk up to 0.6 V forward
        /// of the drain or the source.
        #[test]
        fn jacobians_match_finite_differences(
            drain in -3.0..3.0f64,
            gate in -2.0..3.0f64,
            reverse in -0.6..2.0f64,
            capmod in 0..4u8,
            mobmod in 1..4u8,
            xpart in proptest::sample::select(vec![0.0, 0.5, 1.0]),
            p in proptest::bool::ANY,
        ) {
            let models = format!(r#""capacitance_model": {capmod}, "mobility_model": {mobmod}, "charge_partition": {xpart}"#);
            let parameters = with(SHORT, &models);
            let parameters = if p { p_channel(&parameters) } else { parameters };
            let sign = if p { -1.0 } else { 1.0 };
            let bulk = drain.min(0.0) - reverse;
            assert_jacobians_match::<Bsim3>(
                &parameters,
                &[sign * drain, sign * gate, 0.0, sign * bulk],
            );
            let resistive = with(&parameters, r#""rsh": 20, "drain_squares": 2"#);
            assert_jacobians_match::<Bsim3>(
                &resistive,
                &[sign * drain, sign * gate, 0.0, sign * bulk, sign * 0.95 * drain, sign * 0.01],
            );
        }

        /// A long channel with ngspice's defaults has the derivatives too, its threshold
        /// derived from the doping.
        #[test]
        fn default_jacobians_match_finite_differences(
            drain in -4.0..4.0f64,
            gate in -2.0..4.0f64,
            reverse in 0.0..2.0f64,
        ) {
            let bulk = drain.min(0.0) - reverse;
            assert_jacobians_match::<Bsim3>("{}", &[drain, gate, 0.0, bulk]);
        }
    }

    #[test]
    fn parameter_gradients_are_exact() {
        let binned = r#""lvth0": 2e-8, "wvth0": -1e-8, "pvth0": 1e-15, "lu0": 1e-9,
            "wua": 1e-16, "pk2": 1e-15, "lrdsw": 1e-5, "wvsat": 1e-3, "lcgsl": 1e-17,
            "nominal_temperature": 290, "kt1l": 1e-9, "ute": -1.3, "prt": 20,
            "tpb": 1e-3, "tcj": 1e-3, "xti": 2.5, "rsh": 15"#;
        // Below saturation the substrate current falls to 1e-19 A, whose derivatives
        // central differences no longer resolve: it is checked in saturation alone.
        let substrate = with(SHORT, binned);
        let parameters = substrate.replace(r#""alpha0": 1e-6, "#, "");
        let derived = r#"{"width": 4e-6, "length": 1e-6, "nsub": 2e22, "xt": 1.2e-7,
            "lint": 2e-8, "wint": 3e-8, "ll": 1e-13, "vfb": -2, "dlc": 3e-8, "capacitance_model": 2, "mobility_model": 2,
            "nominal_temperature": 290, "drain_area": 4e-12, "source_area": 4e-12}"#;
        for [drain, gate, bulk] in [
            [2.0, 1.5, -0.5],
            [0.3, 2.0, -0.2],
            [-1.0, 1.2, -1.5],
            [2.0, 0.3, -0.1],
        ] {
            // The internal nodes away from where their currents balance, which would leave
            // differences of their residuals only rounding.
            let inner = [drain, gate, 0.0, bulk, 0.8 * drain, 0.1];
            assert_parameter_gradients_match::<Bsim3>(&parameters, &inner);
            assert_parameter_gradients_match::<Bsim3>(derived, &[drain, gate, 0.0, bulk]);
        }
        assert_parameter_gradients_match::<Bsim3>(&substrate, &[2.5, 1.5, 0.0, -0.5, 2.2, 0.1]);
    }

    #[test]
    fn a_p_channel_mirrors_an_n_channel() {
        for capmod in 0..4 {
            let n = with(SHORT, &format!(r#""capacitance_model": {capmod}"#));
            for x in [
                [2.0, 1.5, 0.0, -0.5],
                [-1.0, 2.0, 0.0, -1.5],
                [1.0, 0.3, 0.0, 0.2],
            ] {
                assert_mirrors(&Differentiable::<Bsim3>::new(), &n, &x);
            }
        }
    }

    #[test]
    fn swapping_the_drain_and_the_source_reverses_the_channel() {
        let symmetric = SHORT
            .replace(r#""source_area": 1.2e-11"#, r#""source_area": 1e-11"#)
            .replace(
                r#""source_perimeter": 8e-6"#,
                r#""source_perimeter": 22e-6"#,
            )
            .replace(r#""cgsl": 1e-10"#, r#""cgsl": 1.2e-10"#);
        let mut bench = Bench::new::<Bsim3>(&symmetric);
        let forward = bench.settle(&[1.5, 2.0, 0.0, -0.5]);
        let reversed = bench.settle(&[0.0, 2.0, 1.5, -0.5]);
        assert_close(&[forward.residual[0]], &[reversed.residual[2]]);
        assert_close(
            &[forward.charge[0], forward.charge[1]],
            &[reversed.charge[2], reversed.charge[1]],
        );
    }

    #[test]
    fn each_dependence_bins_with_the_reciprocal_of_its_extent() {
        // With no length or width lost, the effective channel is the drawn one.
        let (length, width) = (0.5e-6, 2e-6);
        let binned = format!(
            r#"{{"width": {width}, "length": {length}, "vth0": 0.5, "lvth0": 2e-8,
                "wvth0": -3e-8, "pvth0": 1e-14, "u0": 0.04, "lu0": 1e-9, "wu0": -2e-9}}"#
        );
        let vth0 = 0.5 + 2e-8 / length - 3e-8 / width + 1e-14 / (length * width);
        let u0 = 0.04 + 1e-9 / length - 2e-9 / width;
        let plain =
            format!(r#"{{"width": {width}, "length": {length}, "vth0": {vth0}, "u0": {u0}}}"#);
        for x in [[2.0, 1.5, -0.5], [0.2, 1.0, 0.0]] {
            let (binned, plain) = (drain_current(&binned, x), drain_current(&plain, x));
            assert!(
                (binned - plain).abs() <= 1e-13 * plain.abs(),
                "{binned} {plain}"
            );
        }
    }

    #[test]
    fn rejects_what_ngspice_stops_on() {
        let error = |parameters: &str| Bench::try_new::<Bsim3>(parameters).unwrap_err().0;
        assert_eq!(error(r#"{"tox": 0}"#), "tox must be positive, not 0.0");
        assert_eq!(
            error(r#"{"length": 1e-7, "lint": 6e-8}"#),
            "the effective length must be positive, not -1.9999999999999994e-8"
        );
        assert_eq!(
            error(r#"{"length": 1e-6, "nlx": -2e-6}"#),
            "nlx, -2e-6, must be at least the negative of the effective length, 1e-6"
        );
        assert_eq!(
            error(r#"{"dvt1": -0.1}"#),
            "dvt1 must be at least 0, not -0.1"
        );
        assert_eq!(error(r#"{"pclm": 0}"#), "pclm must be positive, not 0.0");
        assert_eq!(
            error(r#"{"mj": 1}"#),
            "mj must be at least 0 and below 1, not 1.0"
        );
        assert_eq!(
            error(r#"{"vsat": 1e4, "at": 1e6, "nominal_temperature": 250}"#),
            "vsat at the temperature must be positive, not -190599.99999999988"
        );
        assert_eq!(
            error(r#"{"mobility_model": 4}"#),
            "BSIM3's mobility_model is 1, 2 or 3, not 4"
        );
    }

    #[test]
    fn clamps_what_ngspice_clamps() {
        let x = [2.0, 1.5, -0.5];
        let current = |parameters: &str| drain_current(parameters, x);
        let charge = |parameters: &str| {
            Bench::new::<Bsim3>(parameters)
                .settle(&[0.0, 0.0, 0.0, -1.0])
                .charge[3]
        };
        // A built-in potential below 0.1 V is raised to it.
        let low = r#"{"pb": 0.05, "drain_area": 1e-11, "source_area": 1e-11}"#;
        let raised = r#"{"pb": 0.1, "drain_area": 1e-11, "source_area": 1e-11}"#;
        assert_eq!(charge(low), charge(raised));
        // With `paramchk`, an `a2` above 1 is 1, and `a1` then 0.
        let checked = r#"{"check_parameters": true, "a1": 0.2, "a2": 1.5, "rdsw": -50}"#;
        assert_eq!(current(checked), current(r#"{"a1": 0, "a2": 1}"#));
        assert_ne!(
            current(r#"{"a1": 0.2, "a2": 1.5}"#),
            current(r#"{"a1": 0, "a2": 1}"#)
        );
    }

    #[test]
    fn unknowns_follow_the_sheet_resistance() {
        let bench = Bench::new::<Bsim3>(r#"{"rsh": 10, "source_squares": 0.05}"#);
        assert_eq!(
            bench.unknowns(),
            [
                "n0",
                "n1",
                "n2",
                "n3",
                "device:drain",
                "device:source",
                "device:source_current"
            ]
        );
        assert_eq!(Bench::new::<Bsim3>("{}").unknowns().len(), 4);
    }

    #[test]
    fn reports_its_threshold_and_saturation_voltages() {
        let mut n = Bench::new::<Bsim3>(SHORT);
        let mut p = Bench::new::<Bsim3>(&p_channel(SHORT));
        n.settle(&[2.0, 1.5, 0.0, -0.5]);
        p.settle(&[-2.0, -1.5, 0.0, 0.5]);
        let (n, p) = (n.variables(), p.variables());
        assert!(n[0] > 0.3 && n[0] < 1.0 && n[1] > 0.0, "{n:?}");
        // The threshold and saturation voltages mirror; the region and the margins, in an
        // n-channel's sign, are the same.
        let polar = [true, true, false, false, false];
        let mirrored: Vec<f64> = p
            .iter()
            .zip(polar)
            .map(|(&value, polar)| if polar { -value } else { value })
            .collect();
        assert_close(&n, &mirrored);
    }

    #[test]
    fn its_channel_makes_the_noise_of_its_noise_model() {
        let x = [2.0, 1.5, 0.0, -0.5];
        // Without a substrate current, the drain's is the channel's.
        let quiet = SHORT
            .replace(r#""alpha0": 1e-6"#, r#""alpha0": 0"#)
            .replace(r#""jsw": 1e-10"#, r#""jsw": 0, "js": 0"#);
        let noisy = with(&quiet, r#""kf": 1e-26, "af": 1.2, "ef": 0.9"#);
        let mut bench = Bench::new::<Bsim3>(&noisy);
        let evaluation = bench.settle(&x);
        let current = evaluation.residual[0];
        // The derivative of the current with respect to lowering the source, gm + gds +
        // gmbs.
        let lowered = -evaluation.conductance[0][2];
        let sources = bench.noise(&x);
        let leff = 0.35e-6;
        let cox = EPSOX / 7.6e-9;
        assert_close(
            &[
                source(&sources, "channel thermal").density(1.0),
                source(&sources, "flicker").density(10.0),
            ],
            &[
                thermal(2.0 / 3.0 * lowered),
                1e-26 * current.powf(1.2) / (leff * leff * cox) / 10f64.powf(0.9),
            ],
        );
        for noimod in 2..=6 {
            let parameters = with(&noisy, &format!(r#""noise_model": {noimod}"#));
            let sources = Bench::new::<Bsim3>(&parameters).noise(&x);
            let flicker = source(&sources, "flicker");
            assert!(
                flicker.density(1.0) > 0.0 && (flicker.exponent - 0.9).abs() < f64::EPSILON,
                "{noimod}"
            );
            assert!(
                source(&sources, "channel thermal").density(1.0) > 0.0,
                "{noimod}"
            );
        }
    }

    #[test]
    fn a_heated_transistor_is_the_transistor_at_its_temperature() {
        let model = crate::Thermal::<Bsim3>::new();
        // Without the substrate current, whose derivatives below saturation central
        // differences no longer resolve.
        let smooth = SHORT.replace(r#""alpha0": 1e-6, "#, "");
        for every in [smooth.clone(), p_channel(&smooth)] {
            let sign = if every.contains(r#""p""#) { -1.0 } else { 1.0 };
            let heated = every.replace('{', r#"{"thermal": true, "#);
            let mut bench =
                Bench::from_model(&model, &heated, REFERENCE_TEMPERATURE).expect("valid");
            for rise in [0.0, 40.0] {
                let mut plain = Bench::at::<Bsim3>(&every, REFERENCE_TEMPERATURE + rise);
                let [d, g, b] = [2.0, 1.5, -0.5].map(|v| sign * v);
                let hot = bench.settle(&[d, g, 0.0, b, rise]);
                let reference = plain.settle(&[d, g, 0.0, b]);
                assert_close(&hot.residual[..4], &reference.residual);
            }
            for [d, g, b, rise] in [[2.0, 1.5, -0.5, 20.0], [0.3, 1.0, -0.2, 60.0]] {
                let x = [sign * d, sign * g, 0.0, sign * b, rise];
                crate::testing::assert_model_jacobians_match(&model, &heated, &x);
                crate::testing::assert_model_parameter_gradients_match(&model, &heated, &x, None);
            }
        }
    }

    /// A transistor is in weak inversion below its threshold, linear above it with its
    /// drain below its saturation voltage, and saturated past it, each region switching
    /// where its margin vanishes, the drain and the source swapping where the drain lies
    /// below.
    #[test]
    fn reports_its_region_and_margins() {
        let mut n = Bench::new::<Bsim3>("{}");
        // The drain, the gate, the source and the bulk, about a threshold of 1.76 V.
        for (x, region) in [
            ([1.0, 0.2, 0.0, 0.0], "weak_inversion"),
            ([0.05, 2.0, 0.0, 0.0], "linear"),
            ([3.0, 3.0, 0.0, 0.0], "saturated"),
            ([0.0, 2.0, 0.05, 0.0], "linear"),
        ] {
            n.settle(&x);
            let (found, margins) = n.region();
            assert_eq!(found, region, "{x:?}");
            // Without a doped poly gate, `Vgs_eff` is the gate's voltage over the lower of
            // the drain and the source, which acts as the source.
            let [drain, gate, source, _] = x;
            let threshold = n.variable("threshold_voltage");
            let saturation = n.variable("saturation_voltage");
            let expected = [
                gate - source.min(drain) - threshold,
                (drain - source).abs() - saturation,
            ];
            assert_close(&margins, &expected);
        }
        let switches = [
            ([3.0, 0.2, 0.0, 0.0], [3.0, 3.0, 0.0, 0.0], OVERDRIVE),
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

    #[test]
    fn saturation_voltage_stays_smooth_with_zero_rds_as_lambda_nears_one() {
        let parameters = SHORT
            .replace(r#""rdsw": 600"#, r#""rdsw": 0"#)
            .replace(r#""alpha0": 1e-6, "#, "");
        assert_ne!(parameters, SHORT);
        let x = [-1.9, 1.7, 0.0, -1.95];
        assert_jacobians_match::<Bsim3>(&parameters, &x);
        let mut bench = Bench::new::<Bsim3>(&parameters);
        let step = 1e-8;
        let mut values = Vec::new();
        for offset in -2..=2 {
            let shifted = [x[0], x[1], f64::from(offset) * step, x[3]];
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
            "saturation_voltage slopes jittered by {spread:e}: {slopes:?}"
        );
    }

    /// Evaluates `bench` at `x` with step limiting disabled and junction leakage `gmin`.
    fn unlimited(bench: &mut Bench, x: &[f64], gmin: f64) -> Evaluation {
        let mut input = Input::new(x, None, gmin);
        input.limiting = false;
        bench.evaluate_with(&input)
    }

    /// Largest absolute entry in a dense square matrix.
    fn matrix_max(matrix: &[Vec<f64>]) -> f64 {
        matrix
            .iter()
            .flatten()
            .fold(0.0_f64, |most, &entry| most.max(entry.abs()))
    }

    /// Checks that two 4-element vectors agree under the drain-source swap `[2, 1, 0, 3]`
    /// within `1024 · f64::EPSILON` of `magnitude` (Higham 2002, §3.1, matching
    /// `testing::laws::ROUNDING`).
    fn assert_vector_swaps(first: &[f64], second: &[f64], magnitude: f64, label: &str) {
        const SWAP: [usize; 4] = [2, 1, 0, 3];
        const ROUNDING: f64 = 1024.0;
        let mut worst = 0.0_f64;
        for (row, &swapped_row) in SWAP.iter().enumerate() {
            worst = worst.max((first[row] - second[swapped_row]).abs());
        }
        assert!(
            worst <= ROUNDING * f64::EPSILON * magnitude,
            "{label} differed by {worst:e} of {magnitude:e}"
        );
    }

    /// Largest difference between two 4x4 matrices under the drain-source swap `[2, 1, 0, 3]`.
    fn max_matrix_swap_difference(first: &[Vec<f64>], second: &[Vec<f64>]) -> f64 {
        const SWAP: [usize; 4] = [2, 1, 0, 3];
        let mut worst = 0.0_f64;
        for (row, &swapped_row) in SWAP.iter().enumerate() {
            for (column, &swapped_column) in SWAP.iter().enumerate() {
                worst = worst.max((first[row][column] - second[swapped_row][swapped_column]).abs());
            }
        }
        worst
    }

    /// Checks that two 4x4 matrices agree under the drain-source swap `[2, 1, 0, 3]`
    /// within `1024 · f64::EPSILON` of their largest entry (Higham 2002, §3.1, matching
    /// `testing::laws::ROUNDING`).
    fn assert_matrix_swaps(first: &[Vec<f64>], second: &[Vec<f64>], label: &str) {
        const ROUNDING: f64 = 1024.0;
        let magnitude = matrix_max(first).max(matrix_max(second));
        let worst = max_matrix_swap_difference(first, second);
        assert!(
            worst <= ROUNDING * f64::EPSILON * magnitude,
            "{label} differed by {worst:e} of {magnitude:e}"
        );
    }

    /// Charge term magnitude bounding rounding error, matching `testing::laws::terms`.
    fn charge_scale(evaluation: &Evaluation, x: &[f64]) -> f64 {
        evaluation
            .charge
            .iter()
            .zip(&evaluation.capacitance)
            .map(|(&charge, row)| {
                let linear: f64 = row
                    .iter()
                    .zip(x)
                    .map(|(&entry, &voltage)| (entry * voltage).abs())
                    .sum();
                charge.abs() + linear
            })
            .fold(0.0_f64, f64::max)
    }

    /// Derived process constants of the default 5 um by 5 um BSIM3 card at 300.15 K,
    /// matching `prepare.rs`.
    struct DefaultBsim3Constants {
        oxide_gate_capacitance: f64,
        surface_potential: f64,
        body_factor_one: f64,
        bulk_clamp_voltage: f64,
        depletion_depth_zero: f64,
        bulk_cv_factor: f64,
    }

    impl DefaultBsim3Constants {
        fn compute() -> Self {
            const INTRINSIC: f64 = 1.45e10;
            const OXIDE_THICKNESS: f64 = 1.5e-8;
            const CHANNEL_LENGTH: f64 = 5e-6;
            const CHANNEL_WIDTH: f64 = 5e-6;
            const CHANNEL_DOPING: f64 = 1.7e17;
            const SUBSTRATE_DOPING: f64 = 6.0e16;
            const DOPING_DEPTH: f64 = 1.55e-7;
            const MAX_BULK_BIAS: f64 = -3.0;

            let nominal = REFERENCE_TEMPERATURE;
            let thermal_volts = nominal * KBOQ;
            let band_gap = -(nominal * nominal * 7.02e-4 / (nominal + 1108.0)) + 1.16;
            let intrinsic = INTRINSIC * (-(band_gap / (thermal_volts * 2.0)) + 21.556_598_1).exp();
            let oxide_per_area = EPSOX / OXIDE_THICKNESS;
            let surface_potential = (CHANNEL_DOPING / intrinsic).ln() * thermal_volts * 2.0;
            let root_potential = surface_potential.sqrt();
            let depletion_depth_zero =
                (2.0 * EPSSI / (CHANNEL_DOPING * (Q * 1e6))).sqrt() * root_potential;
            let gamma_surface = 5.753e-12 * CHANNEL_DOPING.sqrt() / oxide_per_area;
            let gamma_bulk = 5.753e-12 * SUBSTRATE_DOPING.sqrt() / oxide_per_area;
            let transition =
                surface_potential - 7.7348e-4 * CHANNEL_DOPING * DOPING_DEPTH * DOPING_DEPTH;
            let body_factor_two = (gamma_surface - gamma_bulk)
                * ((surface_potential - transition).sqrt() - root_potential)
                / (2.0
                    * ((surface_potential * (surface_potential - MAX_BULK_BIAS)).sqrt()
                        - surface_potential)
                    + MAX_BULK_BIAS);
            let body_factor_one =
                gamma_bulk - 2.0 * body_factor_two * (surface_potential - MAX_BULK_BIAS).sqrt();
            let ratio = body_factor_one * 0.5 / body_factor_two;
            let bulk_clamp_voltage = (surface_potential - ratio * ratio) * 0.9;
            Self {
                oxide_gate_capacitance: oxide_per_area * CHANNEL_WIDTH * CHANNEL_LENGTH,
                surface_potential,
                body_factor_one,
                bulk_clamp_voltage,
                depletion_depth_zero,
                bulk_cv_factor: (1e-7 / CHANNEL_LENGTH).powf(0.6) + 1.0,
            }
        }

        /// Analytical `C[b, d] - C[b, s]` jump in strong inversion at `vds = 0` for `capmod`
        /// 0, `-CoxWL * ((AbulkCV - 1) + k1 * d(sqrt_phis)/d(vbs))`, from `b3ld.c`.
        fn strong_inversion_bulk_jump(&self, bulk_voltage: f64) -> f64 {
            const CHANNEL_LENGTH: f64 = 5e-6;
            const JUNCTION_DEPTH: f64 = 1.5e-7;
            const BULK_CURVATURE: f64 = -0.047;

            let shifted = bulk_voltage - self.bulk_clamp_voltage - 0.001;
            let root = (shifted * shifted - 0.004 * self.bulk_clamp_voltage).sqrt();
            let denominator = root + (bulk_voltage - self.bulk_clamp_voltage + 0.001);
            let effective_bulk = bulk_voltage - 0.002 * bulk_voltage / denominator;
            let bulk_slope = f64::midpoint(1.0, shifted / root);
            let surface_root = (self.surface_potential - effective_bulk).sqrt();
            let surface_root_slope = (-0.5 / surface_root) * bulk_slope;
            let depletion_depth =
                surface_root * (self.depletion_depth_zero / self.surface_potential.sqrt());
            let length_term =
                CHANNEL_LENGTH / (2.0 * (depletion_depth * JUNCTION_DEPTH).sqrt() + CHANNEL_LENGTH);
            let raw_bulk_factor = (self.body_factor_one * 0.5 / surface_root) * length_term + 1.0;
            let capacitance_bulk_factor =
                (raw_bulk_factor / (1.0 + effective_bulk * BULK_CURVATURE)) * self.bulk_cv_factor;
            -self.oxide_gate_capacitance
                * ((capacitance_bulk_factor - 1.0) + self.body_factor_one * surface_root_slope)
        }
    }

    /// Regression test for two BSIM3 bugs at `vbs = 0`:
    /// 1. `capmod_zero` previously tested `channel.vbseff < 0`, sending `vbs = 0` into
    ///    `phi - channel.phis` and inheriting `channel.vbseff`'s left slope
    ///    `|vbsc| / (|vbsc| + 0.001)` instead of `1` (`1.85e-18 F` / `1.4e-4` error).
    /// 2. `channel::Core::evaluate` previously added `smooth_positive` to `vbsc` (`-6.73 V`),
    ///    cancelling ~12 digits near `vbs = 0` and chattering the `vbseff < vbs` clamp
    ///    within `~6e-12 V` of zero across `capmod` 0, 2, and 3.
    #[test]
    fn capmod_zero_vbseff_is_smooth_and_symmetric_at_zero_bulk() {
        const GMIN: f64 = 1e-12;
        const ROUNDING: f64 = 1024.0;
        let card = r#"{"width": 10e-6, "length": 1e-6, "capacitance_model": 0}"#;
        let mut bench = Bench::new::<Bsim3>(card);
        let at_zero = unlimited(&mut bench, &[0.0, 1.2, 0.0, 0.0], GMIN);
        assert_matrix_swaps(
            &at_zero.conductance,
            &at_zero.conductance,
            "10u x 1u capmod 0 conductance at vbs = 0",
        );
        assert_matrix_swaps(
            &at_zero.capacitance,
            &at_zero.capacitance,
            "10u x 1u capmod 0 capacitance at vbs = 0",
        );

        // One-sided finite differences of `Q_g` at `vbs = 0` with `h = 1e-6 V`:
        // truncation error `0.5 * h * |d^2 Q_g / d V_b^2|` is `1.01e-21 F` on the left
        // and `1.38e-20 F` on the right (`phi * vbs / (phi + vbs)` curvature); the pre-fix
        // slope bug was `1.85e-18 F`.
        let step = 1e-6;
        let below = unlimited(&mut bench, &[0.0, 1.2, 0.0, -step], GMIN);
        let above = unlimited(&mut bench, &[0.0, 1.2, 0.0, step], GMIN);
        let left_difference = (at_zero.charge[1] - below.charge[1]) / step;
        let right_difference = (above.charge[1] - at_zero.charge[1]) / step;
        let analytic = at_zero.capacitance[1][3];
        assert!(
            (analytic - left_difference).abs() < 2e-20,
            "left FD mismatch: analytic={analytic:e}, left={left_difference:e}"
        );
        assert!(
            (analytic - right_difference).abs() < 5e-20,
            "right FD mismatch: analytic={analytic:e}, right={right_difference:e}"
        );

        let mut default_bench = Bench::new::<Bsim3>(r#"{"capacitance_model": 0}"#);
        for point in [[0.0, 0.0, 0.0, 0.0], [0.0, -2.0, 0.0, 0.0]] {
            let evaluation = unlimited(&mut default_bench, &point, GMIN);
            assert_matrix_swaps(
                &evaluation.conductance,
                &evaluation.conductance,
                "default capmod 0 conductance",
            );
            assert_matrix_swaps(
                &evaluation.capacitance,
                &evaluation.capacitance,
                "default capmod 0 capacitance",
            );
        }

        // Branch-chatter check across `vbs = 0` on `10u x 1u, Vg = 1.2 V`.
        let sweep = [-1e-9, -1e-12, -1e-15, 0.0, 1e-15, 1e-12, 1e-9];
        for bulk_voltage in sweep {
            let evaluation = unlimited(&mut bench, &[0.0, 1.2, 0.0, bulk_voltage], GMIN);
            let maximum_capacitance = matrix_max(&evaluation.capacitance);
            let difference = (evaluation.capacitance[1][0] - evaluation.capacitance[1][2]).abs();
            let bound = if bulk_voltage <= 0.0 {
                ROUNDING * f64::EPSILON * maximum_capacitance
            } else {
                // Forward body bias slope defect is `1 - (phi/(phi+vbs))^2 <= 2.5*vbs/0.8`.
                ROUNDING * f64::EPSILON * maximum_capacitance
                    + 2.5 * bulk_voltage * maximum_capacitance / 0.8
            };
            assert!(
                difference <= bound,
                "capmod 0 at vbs={bulk_voltage:e}: diff={difference:e} > {bound:e}"
            );
        }
        for model_index in [2_u8, 3_u8] {
            let parameters = format!(
                r#"{{"width": 10e-6, "length": 1e-6, "capacitance_model": {model_index}}}"#
            );
            let mut sweep_bench = Bench::new::<Bsim3>(&parameters);
            for bulk_voltage in sweep {
                let evaluation = unlimited(&mut sweep_bench, &[0.0, 1.2, 0.0, bulk_voltage], GMIN);
                let difference = evaluation.capacitance[1][0] - evaluation.capacitance[1][2];
                if bulk_voltage <= 0.0 {
                    // `b3ld.c` keeps `1 - |vbsc|/(|vbsc|+0.001)` (`1.92e-18` / `1.79e-18`).
                    assert!(
                        difference > 1.5e-18,
                        "capmod {model_index} chattered at vbs={bulk_voltage:e}: diff={difference:e}"
                    );
                } else {
                    // For `vbs > 0`, `vbs` replaces the smooth floor; max asymmetry at `1e-9`
                    // is `3.13e-22 F`, far below the pre-fix `1.79e-18 F` chatter.
                    assert!(
                        difference.abs() < 1e-20,
                        "capmod {model_index} chattered at vbs={bulk_voltage:e}: diff={difference:e}"
                    );
                }
            }
        }
    }

    /// Regression test for `effective_drain` catastrophic cancellation at small `vds`:
    /// previously `vdsat - (t1 + t2) * 0.5` lost ~8 digits at `vds = 1e-9 V` (failing swap
    /// symmetry by `8.58e-24 A`, `37,886` eps) and lost 5 to 14 digits on `Ids(vds) / vds`
    /// at `1e-12` to `1e-15 V`.
    #[test]
    fn effective_drain_keeps_precision_at_small_vds() {
        const GMIN: f64 = 1e-12;
        let mut bench = Bench::new::<Bsim3>(r#"{"capacitance_model": 0}"#);
        let forward_point = [1e-9, 2.0, 0.0, -0.5];
        let swapped_point = [0.0, 2.0, 1e-9, -0.5];
        let forward = unlimited(&mut bench, &forward_point, GMIN);
        let reversed = unlimited(&mut bench, &swapped_point, GMIN);
        let residual_scale = forward
            .scale
            .iter()
            .chain(&reversed.scale)
            .copied()
            .fold(0.0_f64, f64::max);
        assert_vector_swaps(
            &forward.residual,
            &reversed.residual,
            residual_scale,
            "small-vds residual",
        );
        let charge_magnitude =
            charge_scale(&forward, &forward_point).max(charge_scale(&reversed, &swapped_point));
        assert_vector_swaps(
            &forward.charge,
            &reversed.charge,
            charge_magnitude,
            "small-vds charge",
        );
        assert_matrix_swaps(
            &forward.conductance,
            &reversed.conductance,
            "small-vds conductance",
        );
        assert_matrix_swaps(
            &forward.capacitance,
            &reversed.capacitance,
            "small-vds capacitance",
        );

        let no_junction = r#"{"drain_area": 1e-12, "source_area": 1e-12, "js": 0, "jsw": 0}"#;
        let mut quiet = Bench::new::<Bsim3>(no_junction);
        for (gate_voltage, bulk_voltage) in [(3.0, 0.0), (2.0, -0.5)] {
            let zero_slope = unlimited(&mut quiet, &[0.0, gate_voltage, 0.0, bulk_voltage], 0.0)
                .conductance[0][0];
            assert!(zero_slope > 0.0, "expected positive gds(0)");
            for drain_voltage in [1e-15, 1e-12] {
                let secant = unlimited(
                    &mut quiet,
                    &[drain_voltage, gate_voltage, 0.0, bulk_voltage],
                    0.0,
                )
                .residual[0]
                    / drain_voltage;
                // Quadratic error `O(vds / vdsat)` is at most `~3e-11` at `vds = 1e-12`
                // (`vdsat ~ 0.037 V`); pre-fix cancellation error was `~1e-5` to `1.5e-2`.
                assert!(
                    (secant - zero_slope).abs() <= 1e-9 * zero_slope,
                    "vg={gate_voltage}, vb={bulk_voltage}, vds={drain_voltage:e}: secant={secant:e}, gds0={zero_slope:e}"
                );
            }
        }
    }

    /// Asserts that `capmod` 0 in strong inversion has a symmetric gate row and matches the
    /// exact analytical bulk and drain-source capacitance jumps of `b3ld.c` lines 1330 to
    /// 1582 across all three charge partitions.
    fn assert_capmod_zero_strong_inversion_jumps(constants: &DefaultBsim3Constants, gmin: f64) {
        const ROUNDING: f64 = 1024.0;
        for bulk_voltage in [0.0, -0.5] {
            let expected_bulk = constants.strong_inversion_bulk_jump(bulk_voltage);
            let expected_drain = -0.5 * expected_bulk;
            for partition in [0.0, 0.5, 1.0] {
                let card =
                    format!(r#"{{"capacitance_model": 0, "charge_partition": {partition}}}"#);
                let mut bench = Bench::new::<Bsim3>(&card);
                let evaluation = unlimited(&mut bench, &[0.0, 3.0, 0.0, bulk_voltage], gmin);
                let tolerance = ROUNDING * f64::EPSILON * matrix_max(&evaluation.capacitance);
                assert!(
                    (evaluation.capacitance[1][0] - evaluation.capacitance[1][2]).abs()
                        <= tolerance,
                    "capmod 0 gate row asymmetric at vb={bulk_voltage}, xpart={partition}"
                );
                let bulk_jump = evaluation.capacitance[3][0] - evaluation.capacitance[3][2];
                assert!(
                    (bulk_jump - expected_bulk).abs() <= tolerance,
                    "bulk jump {bulk_jump:e} != expected {expected_bulk:e}"
                );
                let drain_jump = evaluation.capacitance[0][0] - evaluation.capacitance[2][2];
                assert!(
                    (drain_jump - expected_drain).abs() <= tolerance,
                    "drain jump {drain_jump:e} != expected {expected_drain:e}"
                );
            }
        }
    }

    /// Verifies that at `vds = 0`, currents, charges (`Q_d == Q_s`), and conductances swap
    /// cleanly across all four `capmod`s, and bounds the exact remaining capacitance
    /// asymmetries inherent in Berkeley BSIM3v3.3 (`b3ld.c`).
    #[test]
    fn swap_symmetry_at_zero_vds_matches_bsim3_equations() {
        const GMIN: f64 = 1e-12;
        const ROUNDING: f64 = 1024.0;
        let constants = DefaultBsim3Constants::compute();

        // Currents, charges, and conductances swap cleanly at `vds = 0` for all `capmod`s.
        let biases = [
            (-2.0, 0.0),
            (0.0, 0.0),
            (1.2, 0.0),
            (3.0, 0.0),
            (3.0, -0.5),
            (-2.0, 0.1),
        ];
        for model_index in 0..=3_u8 {
            let card = format!(r#"{{"capacitance_model": {model_index}}}"#);
            let mut bench = Bench::new::<Bsim3>(&card);
            for (gate_voltage, bulk_voltage) in biases {
                let point = [0.0, gate_voltage, 0.0, bulk_voltage];
                let evaluation = unlimited(&mut bench, &point, GMIN);
                let residual_scale = evaluation.scale.iter().copied().fold(0.0_f64, f64::max);
                assert_vector_swaps(
                    &evaluation.residual,
                    &evaluation.residual,
                    residual_scale,
                    "zero-vds residual",
                );
                assert_vector_swaps(
                    &evaluation.charge,
                    &evaluation.charge,
                    charge_scale(&evaluation, &point),
                    "zero-vds charge",
                );
                assert_matrix_swaps(
                    &evaluation.conductance,
                    &evaluation.conductance,
                    "zero-vds conductance",
                );
            }
        }

        assert_capmod_zero_strong_inversion_jumps(&constants, GMIN);

        // `capmod` 0 forward body bias (`b3ld.c` lines 1271 to 1274): `VbseffCV` has slope
        // `(phi / (phi + vbs))^2 < 1`, giving `+1.157e-14 F` (`19.2%`) at `vbs = 0.1 V`.
        let mut zero_bench = Bench::new::<Bsim3>(r#"{"capacitance_model": 0}"#);
        let forward_body = unlimited(&mut zero_bench, &[0.0, -2.0, 0.0, 0.1], GMIN);
        let expected_forward = constants.oxide_gate_capacitance
            * (1.0 - (constants.surface_potential / (constants.surface_potential + 0.1)).powi(2));
        let gate_jump = forward_body.capacitance[1][0] - forward_body.capacitance[1][2];
        assert!(
            (gate_jump - expected_forward).abs()
                <= ROUNDING * f64::EPSILON * matrix_max(&forward_body.capacitance),
            "forward-body jump {gate_jump:e} != {expected_forward:e}"
        );

        // `capmod` 1 (`b3ld.c` lines 1702 to 1788): linear branch remains active at `vds = 0`
        // below threshold (`1.474x` of `max|C|` on `10u x 1u` at `vg = 1.2 V`).
        let mut model_one_bench =
            Bench::new::<Bsim3>(r#"{"width": 10e-6, "length": 1e-6, "capacitance_model": 1}"#);
        let model_one = unlimited(&mut model_one_bench, &[0.0, 1.2, 0.0, 0.0], GMIN);
        let model_one_max = matrix_max(&model_one.capacitance);
        assert!(model_one.capacitance[1][0] < -1.4e-14 && model_one.capacitance[1][2] > 8.5e-15);
        assert!(
            (model_one.capacitance[1][0] - model_one.capacitance[1][2]).abs() / model_one_max > 1.1
        );
        assert!(
            max_matrix_swap_difference(&model_one.capacitance, &model_one.capacitance)
                / model_one_max
                > 1.4
        );

        // `capmod` 2 and 3 (`b3ld.c` lines 1585 to 2239): `VbseffCV = Vbseff` has slope
        // `1 - 1.485e-4` at `vbs = 0`, giving `1.391e-4` (`capmod` 2) and `1.385e-4` (`3`).
        for model_index in [2_u8, 3_u8] {
            let mut bench =
                Bench::new::<Bsim3>(&format!(r#"{{"capacitance_model": {model_index}}}"#));
            let evaluation = unlimited(&mut bench, &[0.0, 0.0, 0.0, 0.0], GMIN);
            let ratio = (evaluation.capacitance[1][0] - evaluation.capacitance[1][2])
                / matrix_max(&evaluation.capacitance);
            assert!(
                (1.3e-4..=1.5e-4).contains(&ratio),
                "capmod {model_index} ratio {ratio:e} outside [1.3e-4, 1.5e-4]"
            );
        }
    }

    /// Verifies that deep-subthreshold `capmod` 3 preserves the exact logarithmic slope
    /// `1 / (n * vtm)` of inversion charge using `ln_1p` and rationalized `smooth_positive`
    /// instead of collapsing under `1 + x` and `t0 + sqrt(t0^2 + spread)` cancellation.
    #[test]
    fn capmod_three_preserves_subthreshold_inversion_slope() {
        let parameters = Bsim3Parameters::<f64>::default();
        let prepared = Bsim3::prepare(&parameters, REFERENCE_TEMPERATURE).expect("valid");
        let probe = prepared.core.evaluate(Dual::variables([0.0, 1.0, 0.0]));
        let n_vtm = probe.n.value * prepared.core.vtm;
        let gate = probe.vth.value - 28.0 * n_vtm;
        let inversion = |vgs: f64| {
            let variables = Dual::variables([vgs, 1.0, 0.0]);
            let channel = prepared.core.evaluate(variables);
            prepared
                .core
                .intrinsic(variables[1], variables[2], &channel)
                .inversion
        };
        let step = 1e-6;
        let center = inversion(gate);
        assert!(
            center < 0.0,
            "expected negative inversion charge, got {center:e}"
        );
        let slope = (inversion(gate + step) - inversion(gate - step)) / (2.0 * step * center);
        let expected = 1.0 / n_vtm;
        assert!(
            ((slope - expected) / expected).abs() < 1e-4,
            "capmod 3 deep-subthreshold log slope {slope:e} differed from {expected:e}"
        );
    }
}
