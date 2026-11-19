//! Shared physics, smoothing functions, charge laws and preparation helpers of BSIM3v3.3
//! and BSIM4.5, as ngspice 47 implements them in `src/spicelib/devices/bsim3/` and
//! `src/spicelib/devices/bsim4v5/`.
//!
//! This module is derived from BSIM3v3.3.0 and BSIM4.5.0, developed by the UC Berkeley
//! BSIM Research Group: Copyright 2004, 2005 Regents of the University of California.
//! All rights reserved. It is used under the BSIM terms of use, which the repository's
//! `NOTICE` reproduces.

use crate::kernel::dual::Dual;
use crate::kernel::jfet::Channel;
use crate::kernel::region::{OVERDRIVE, SATURATION_MARGIN};
use crate::{Branch, BranchKind, Interface, ParameterError, REGION, Real, Regions};

/// Boltzmann's constant over the elementary charge, in volts per kelvin, as ngspice's
/// `b3temp.c` and `b4v5temp.c` take it.
pub(crate) const KBOQ: f64 = 8.617_087e-5;
/// The permittivity of silicon, in farads per meter, as ngspice's BSIM3 and BSIM4 take
/// it.
pub(crate) const EPSSI: f64 = 1.035_94e-10;
/// The elementary charge, in coulombs, as ngspice's `b3temp.c`, `b3ld.c`, `b4v5temp.c`
/// and `b4v5ld.c` take it.
pub(crate) const Q: f64 = 1.602_19e-19;
/// The intrinsic carrier concentration of silicon at 300.15 K, per cubic centimeter, as
/// ngspice's `b3temp.c` and `b4v5temp.c` take it.
pub(crate) const INTRINSIC: f64 = 1.45e10;
/// `e^34`, where ngspice's `b3ld.c` and `b4v5ld.c` stop their exponentials.
pub(crate) const MAX_EXP: f64 = 5.834_617_425e14;
/// `e^−34`, as ngspice's `b3ld.c` and `b4v5ld.c` write it.
pub(crate) const MIN_EXP: f64 = 1.713_908_431e-15;
/// The argument beyond which ngspice's `b3ld.c` and `b4v5ld.c` stop their exponentials.
pub(crate) const EXP_THRESHOLD: f64 = 34.0;
/// The smoothing of ngspice's `b3ld.c` and `b4v5ld.c` effective voltages of the charges,
/// their `DELTA_1` to `DELTA_4`, in volts.
pub(crate) const DELTA: f64 = 0.02;
/// The elementary charge, in coulombs, as ngspice's `const.h` gives it to `b3noi.c` and
/// to `b4v5ld.c`'s poly gate depletion.
pub(crate) const NOISE_CHARGE: f64 = 1.602_176_620_8e-19;
/// The least argument of ngspice's `b3noi.c` and `b4v5noi.c` logarithms, their
/// `N_MINLOG`.
pub(crate) const N_MINLOG: f64 = 1e-38;
/// Boltzmann's constant, in joules per kelvin, as ngspice's `const.h` gives it,
/// `CONSTboltz`, to `b4v5noi.c` and to its thermal voltage.
pub(crate) const NGSPICE_BOLTZMANN: f64 = 1.380_648_52e-23;
/// The thermal voltage at 27 °C with ngspice's constants, `CONSTvt0`, from which
/// `b3ld.c` and `b4v5ld.c` limit the bulk junctions' steps.
pub(crate) const VT0: f64 = NGSPICE_BOLTZMANN * 300.15 / NOISE_CHARGE;

/// The variables a BSIM transistor reports: its threshold and saturation voltages, in
/// its own polarity, and its region among [`REGIONS`] and its margins.
pub(crate) const VARIABLE_NAMES: [&str; 5] = [
    "threshold_voltage",
    "saturation_voltage",
    REGION,
    OVERDRIVE,
    SATURATION_MARGIN,
];

/// The regions a BSIM transistor's equations distinguish, and its margins to their
/// boundaries: `Vgst = Vgs_eff − Vth` and `Vds − Vdsat`, in an n-channel's sign as
/// [`Regions::margins`] are, with the drain and the source swapped where the drain lies
/// below the source, as the equations swap them.
pub(crate) const REGIONS: Regions = Regions {
    names: &["weak_inversion", "linear", "saturated"],
    margins: &[OVERDRIVE, SATURATION_MARGIN],
};

/// A number and its derivatives with respect to a transistor's gate-source, drain-source
/// and bulk-source voltages.
pub(crate) type D<R> = Dual<R, 3>;

/// The regions of a BSIM transistor's channel, on either side of its threshold and
/// saturation voltages, `Vth` and `Vdsat`, in the order of [`REGIONS`]' names. The
/// equations blend one region into the next through `Vgsteff` and `Vdseff`, and switch
/// at neither.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Region {
    /// Below its threshold, its current falling exponentially.
    WeakInversion,
    /// At or above its threshold, with the drain below saturation.
    Linear,
    /// At or above its threshold, with the drain at or past saturation.
    Saturated,
}

impl Region {
    /// Classifies a channel from its `[overdrive, saturation_margin]` pair.
    pub(crate) fn classify([overdrive, margin]: [f64; 2]) -> Self {
        if overdrive < 0.0 {
            Self::WeakInversion
        } else if margin < 0.0 {
            Self::Linear
        } else {
            Self::Saturated
        }
    }
}

/// How `xpart` shares the channel's charge in saturation: the drain takes none, 40 % or
/// half.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Partition {
    /// `xpart` above 0.5: 0/100.
    None,
    /// `xpart` below 0.5: 40/60.
    Forty,
    /// `xpart` of 0.5: 50/50.
    Half,
}

impl Partition {
    /// Selects the charge partition branch for `xpart`.
    pub(crate) fn from_xpart(xpart: f64) -> Self {
        if xpart > 0.5 {
            Self::None
        } else if xpart < 0.5 {
            Self::Forty
        } else {
            Self::Half
        }
    }
}

/// The critical voltage of a junction of 10 fA at [`VT0`], `VT0 · ln(VT0 / (√2 · 10 fA))`,
/// from which `b3ld.c` and `b4v5ld.c` limit the bulk junctions' steps, their `vcrit`.
pub(crate) fn critical_voltage() -> f64 {
    VT0 * (VT0 / (std::f64::consts::SQRT_2 * 1.0e-14)).ln()
}

/// `n` for an n-channel and `p` for a p-channel.
pub(crate) const fn by_channel(channel: Channel, n: f64, p: f64) -> f64 {
    match channel {
        Channel::N => n,
        Channel::P => p,
    }
}

/// The four-terminal MOSFET interface shared by BSIM3 and BSIM4.
pub(crate) fn interface() -> Interface {
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

/// Orients `[vgs, vds, vbs]` into the frame whose drain lies at or above its source,
/// returning whether the channel is forward (`vds >= 0`) and the frame's voltages.
pub(crate) fn forward_frame<R: Real>(voltages: [R; 3]) -> (bool, [R; 3]) {
    let [gate, drain, bulk] = voltages;
    let forward = drain.value() >= 0.0;
    let frame = if forward {
        voltages
    } else {
        [gate - drain, -drain, bulk - drain]
    };
    (forward, frame)
}

/// `function`, of the gate-source, drain-source and bulk-source voltages of the frame
/// whose source is the drain, with its derivatives with respect to the terminals' own
/// voltages.
pub(crate) fn unswapped<R: Real>(function: D<R>) -> D<R> {
    let [gate, drain, bulk] = function.gradient;
    D {
        value: function.value,
        gradient: [gate, -(gate + drain + bulk), bulk],
    }
}

/// The values of [`VARIABLE_NAMES`] of a channel of `sign`, in its own polarity.
pub(crate) fn variables(
    sign: f64,
    threshold: f64,
    saturation: f64,
    region: Region,
    [overdrive, margin]: [f64; 2],
) -> [f64; 5] {
    [
        sign * threshold,
        sign * saturation,
        f64::from(region as u8),
        overdrive,
        margin,
    ]
}

/// The logarithmic fringing capacitance factor `ln(1 + 0.4 µm / tox)`, shared by
/// `b3set.c` and `b4v5set.c`.
pub(crate) fn fringe_factor<R: Real>(oxide_thickness: R) -> R {
    (R::constant(0.4e-6) / oxide_thickness + 1.0).ln()
}

/// Fails with `message` when `failed` holds, as `b3check.c` and `b4v5check.c` stop on a
/// fatal error.
pub(crate) fn check(failed: bool, message: impl FnOnce() -> String) -> Result<(), ParameterError> {
    if failed {
        Err(ParameterError(message()))
    } else {
        Ok(())
    }
}

/// The reciprocal-free length or width lost, `int + l / L^ln + w / W^wn + wl / (L^ln · W^wn)`.
pub(crate) fn lost<R: Real>(
    base: R,
    [by_length, by_width, by_area]: [R; 3],
    [length, width]: [R; 2],
) -> R {
    base + by_length / length + by_width / width + by_area / (length * width)
}

/// Silicon's band gap at `temperature`, in electronvolts, as `b3temp.c` and `b4v5temp.c`
/// compute it: `1.16 − 7.02e-4 · T² / (T + 1108)`.
pub(crate) fn band_gap<R: Real>(temperature: R) -> R {
    -(temperature * temperature * 7.02e-4 / (temperature + 1108.0)) + 1.16
}

/// Silicon's intrinsic carrier concentration at `nominal_temperature`, per cubic
/// centimeter, given its `band_gap` and `thermal_voltage`, as `b3temp.c` and `b4v5temp.c`
/// compute it.
pub(crate) fn intrinsic_carrier<R: Real>(
    nominal_temperature: R,
    band_gap: R,
    thermal_voltage: R,
) -> R {
    let reference = nominal_temperature / 300.15;
    reference
        * reference.sqrt()
        * INTRINSIC
        * (-(band_gap / (thermal_voltage * 2.0)) + 21.556_598_1).exp()
}

/// Derives `(k1, k2, vbm)` from the surface and bulk body effects when neither `k1` nor
/// `k2` is given, negating a positive `vbm` as `b3temp.c` and `b4v5temp.c` do.
pub(crate) fn derived_body_factors<R: Real>(
    (phi, sqrt_phi): (R, R),
    (doping, substrate_doping, cox): (R, R, R),
    (transition_bias, mut max_bulk_bias, depth): (Option<R>, R, R),
    (surface_gamma, bulk_gamma): (Option<R>, Option<R>),
) -> (R, R, R) {
    let mut transition_bias =
        transition_bias.unwrap_or_else(|| phi - doping * depth * depth * 7.7348e-4);
    if transition_bias.value() > 0.0 {
        transition_bias = -transition_bias;
    }
    if max_bulk_bias.value() > 0.0 {
        max_bulk_bias = -max_bulk_bias;
    }
    let surface_gamma = surface_gamma.unwrap_or_else(|| doping.sqrt() * 5.753e-12 / cox);
    let bulk_gamma = bulk_gamma.unwrap_or_else(|| substrate_doping.sqrt() * 5.753e-12 / cox);
    let t0 = surface_gamma - bulk_gamma;
    let t1 = (phi - transition_bias).sqrt() - sqrt_phi;
    let t2 = (phi * (phi - max_bulk_bias)).sqrt() - phi;
    let k2 = t0 * t1 / (t2 * 2.0 + max_bulk_bias);
    let k1 = bulk_gamma - k2 * (phi - max_bulk_bias).sqrt() * 2.0;
    (k1, k2, max_bulk_bias)
}

/// The most negative effective bulk voltage `vbsc`, clamped to `[-30, -3]` V and at most
/// `max_bulk_bias`, as `b3temp.c` and `b4v5temp.c` compute it.
pub(crate) fn vbsc_clamp<R: Real>(phi: R, k1: R, k2: R, max_bulk_bias: R) -> R {
    let bound = if k2.value() < 0.0 {
        let t0 = k1 * 0.5 / k2;
        let bound = (phi - t0 * t0) * 0.9;
        if bound.value() > -3.0 {
            R::constant(-3.0)
        } else if bound.value() < -30.0 {
            R::constant(-30.0)
        } else {
            bound
        }
    } else {
        R::constant(-30.0)
    };
    if bound.value() > max_bulk_bias.value() {
        max_bulk_bias
    } else {
        bound
    }
}

/// Derives default `[cgso, cgdo, cgbo]` from the oxide capacitance `cox`, junction depth
/// `junction_depth`, `length_offset` (`dlc`), `width_offset` (`dwc` or `wint`), and the
/// lightly-doped overlaps `[source_lightly, drain_lightly]`, as `b3set.c` and `b4v5temp.c`
/// do.
pub(crate) fn default_overlaps<R: Real>(
    [source_given, drain_given, bulk_given]: [Option<R>; 3],
    [source_lightly, drain_lightly]: [R; 2],
    (cox, junction_depth): (R, R),
    (length_offset, width_offset): (Option<R>, R),
) -> [R; 3] {
    let positive_offset = length_offset.filter(|value| value.value() > 0.0);
    let side = |given: Option<R>, lightly: R| {
        given.unwrap_or_else(|| {
            positive_offset.map_or(junction_depth * cox * 0.6, |offset| offset * cox - lightly)
        })
    };
    [
        side(source_given, source_lightly),
        side(drain_given, drain_lightly),
        bulk_given.unwrap_or_else(|| width_offset * cox * 2.0),
    ]
}

/// A junction capacitance scaled linearly with `delta_temperature`, floored at zero, as
/// `b3temp.c` and `b4v5temp.c` compute it.
pub(crate) fn heated_capacitance<R: Real>(
    capacitance: R,
    coefficient: R,
    delta_temperature: R,
) -> R {
    let t0 = coefficient * delta_temperature;
    if t0.value() >= -1.0 {
        capacitance * (t0 + 1.0)
    } else {
        R::constant(0.0)
    }
}

/// A junction built-in potential raised to at least 0.1 V and shifted linearly with
/// `delta_temperature`, floored at 0.01 V, as `b3temp.c` and `b4v5temp.c` compute it.
pub(crate) fn heated_potential<R: Real>(potential: R, coefficient: R, delta_temperature: R) -> R {
    let raised = potential.max(R::constant(0.1));
    (raised - coefficient * delta_temperature).max(R::constant(0.01))
}

/// The clamp of `b3ld.c` and `b4v5ld.c` that keeps a factor `1 + x` from approaching
/// zero: `1 + x` down to `x = −0.9`, then `(17 + 20x) / (0.8 + x)`'s reciprocal, its
/// first form's value and slope continued.
pub(crate) fn reciprocal_above_tenth<R: Real>(x: D<R>) -> D<R> {
    if x.value.value() >= -0.9 {
        (x + 1.0).recip()
    } else {
        (x * 20.0 + 17.0) / (x + 0.8)
    }
}

/// `1 + x`, continued below `x = −0.9` by `(0.8 + x) / (17 + 20x)`, as `b3ld.c` and
/// `b4v5ld.c` keep such a factor positive.
pub(crate) fn factor_above_tenth<R: Real>(x: D<R>, inclusive: bool) -> D<R> {
    let above = if inclusive {
        x.value.value() >= -0.9
    } else {
        x.value.value() > -0.9
    };
    if above {
        x + 1.0
    } else {
        (x + 0.8) / (x * 20.0 + 17.0)
    }
}

/// `x`, continued below 0.1 by `(0.2 − x) / (3 − 20x)`.
pub(crate) fn above_tenth<R: Real>(x: D<R>) -> D<R> {
    if x.value.value() < 0.1 {
        (-x + 0.2) / (-(x * 20.0) + 3.0)
    } else {
        x
    }
}

/// `1 + x`, continued below `x = −0.8` by `(0.6 + x) / (7 + 10x)`, keeping the mobility
/// denominator positive in `b3ld.c` and `b4v5ld.c`.
pub(crate) fn above_fifth<R: Real>(x: D<R>) -> D<R> {
    if x.value.value() >= -0.8 {
        x + 1.0
    } else {
        (x + 0.6) / (x * 10.0 + 7.0)
    }
}

/// `1 + x`, continued below `x = −0.5` by `(1 + 3x) / (3 + 8x)`.
pub(crate) fn above_half<R: Real>(x: D<R>) -> D<R> {
    if x.value.value() >= -0.5 {
        x + 1.0
    } else {
        (x * 3.0 + 1.0) / (x * 8.0 + 3.0)
    }
}

/// `x`, continued below `1e-4` by `(2e-4 − x) / (3 − 2e4 · x)`, keeping the
/// drain-induced barrier lowering factor positive in `b3ld.c` and `b4v5ld.c`.
pub(crate) fn above_dibl_floor<R: Real>(x: D<R>) -> D<R> {
    if x.value.value() < 1.0e-4 {
        (2.0e-4 - x) / (3.0 - x * 2.0e4)
    } else {
        x
    }
}

/// The effective width `width`, continued below 20 nm by
/// `2e-8 · (4e-8 − width) / (6e-8 − 2 · width)`, as `b3ld.c` and `b4v5ld.c` keep it
/// positive.
pub(crate) fn effective_width<R: Real>(width: D<R>) -> D<R> {
    if width.value.value() < 2.0e-8 {
        (4.0e-8 - width) * 2.0e-8 / (6.0e-8 - width * 2.0)
    } else {
        width
    }
}

/// `(x + √(x² + spread)) / 2`, the positive branch of a smooth clamp: written as
/// `(spread / 2) / (√(x² + spread) − x)` where `x < 0` so `x` and its root never cancel.
pub(crate) fn smooth_positive<R: Real>(x: D<R>, spread: D<R>) -> D<R> {
    let root = (x * x + spread).sqrt();
    if x.value.value() >= 0.0 {
        (x + root) * 0.5
    } else {
        (spread * 0.5) / (root - x)
    }
}

/// The smooth lower clamp `bound + (shifted + √(shifted² − 0.004 · bound)) / 2` with
/// `shifted = x − bound − 0.001` and `bound < 0`, shared by `b3ld.c` and `b4v5ld.c`.
///
/// By `(shifted² − 0.004 · bound) − (x − bound + 0.001)² = −0.004 · x`, the curve passes
/// through zero, lies strictly above `x` for `x < 0`, and lies strictly below `x` for
/// `x > 0`. For `x >= bound / 2`, rationalizing it as
/// `x − 0.002 · x / (root + x − bound + 0.001)` prevents `bound` and `smooth_positive`
/// from cancelling near `x = 0` (losing ~12 digits and chattering BSIM3's `vbseff < vbs`
/// clamp within ~6e-12 V of zero); below `bound / 2`, `smooth_positive(shifted, ...) + bound`
/// stays exact through the knee and at large negative `x`.
pub(crate) fn smooth_floor<R: Real>(x: D<R>, bound: R) -> D<R> {
    let shifted = (x - 0.001).plus(-bound);
    let spread = D::constant(bound * -0.004);
    if x.value.value() >= 0.5 * bound.value() {
        let root = (shifted * shifted + spread).sqrt();
        let denominator = root + (x + 0.001).plus(-bound);
        x - (x * 0.002) / denominator
    } else {
        smooth_positive(shifted, spread).plus(bound)
    }
}

/// The gate-source voltage less the poly gate's depletion, shared by `b3ld.c` and
/// `b4v5ld.c` (`BSIM4v5polyDepletion`), with `gate_doping` in cm⁻³,
/// `flat_band_potential = vfb + phi`, oxide capacitance `cox`, and elementary charge
/// `charge` ([`Q`] in BSIM3, [`NOISE_CHARGE`] in BSIM4).
pub(crate) fn poly_depletion<R: Real>(
    gate_source: D<R>,
    gate_doping: R,
    flat_band_potential: R,
    cox: R,
    charge: f64,
) -> D<R> {
    let doping = gate_doping.value();
    if doping > 1.0e18 && doping < 1.0e25 && gate_source.value.value() > flat_band_potential.value()
    {
        let t1 = gate_doping * (1.0e6 * charge * EPSSI) / (cox * cox);
        let t8 = gate_source.plus(-flat_band_potential);
        let t4 = (t8.divide(t1) * 2.0 + 1.0).sqrt();
        let t2 = t8 * 2.0 / (t4 + 1.0);
        let t3 = (t2 * t2 * 0.5).divide(t1);
        let t7 = 1.12 - t3 - 0.05;
        let t5 = 1.12 - smooth_positive(t7, D::number(0.224));
        gate_source - t5
    } else {
        gate_source
    }
}

/// The smooth minimum of `vds >= 0` and `vdsat >= 0` with smoothing `delta >= 0`,
/// `vdsat − (shifted + root) / 2` where `shifted = vdsat − vds − delta` and
/// `root = √(shifted² + 4 · delta · vdsat)`, shared by BSIM3 and BSIM4 for both `Vdseff`
/// and `VdseffCV`.
///
/// Rationalized on both sides of `vds = vdsat + delta`:
/// - `vds <= vdsat + delta`: `2 · vdsat · vds / (vdsat + vds + delta + root)` eliminates
///   cancellation of `vdsat` near `vds = 0` (even when `vdsat < delta`), gives exact zero
///   with zero `gate` and `bulk` derivatives at `vds = 0`, and keeps full precision at
///   small `vds` such as 1e-9 V where the raw difference loses ~8 digits;
/// - `vds > vdsat + delta`: `vdsat · (1 − 2 · delta / (root − shifted))` keeps
///   `2 · delta / (root − shifted) <= 0.5` and preserves full precision on `d/d(vds)` at
///   large `vds`.
pub(crate) fn effective_drain<R: Real>(vds: D<R>, vdsat: D<R>, delta: R) -> D<R> {
    let shifted = vdsat - vds - D::constant(delta);
    let root = (shifted * shifted + vdsat.scale(delta * 4.0)).sqrt();
    if vds.value.value() <= vdsat.value.value() + delta.value() {
        let denominator = (vdsat + vds + root).plus(delta);
        if denominator.value.value() <= 0.0 {
            D::number(0.0)
        } else {
            (vdsat * vds * 2.0) / denominator
        }
    } else {
        let ratio = D::constant(delta * 2.0) / (root - shifted);
        vdsat * (-ratio + 1.0)
    }
}

/// `1 / λ − 1`, the non-saturation factor `λ` of `b3ld.c` and `b4v5ld.c` written in its
/// excess over unity so `1 − λ` and `1 / λ − 1` never cancel as `λ → 1`.
pub(crate) fn non_saturation_excess<R: Real>(a1: R, a2: R, vgsteff: D<R>) -> D<R> {
    let slope = a1.value();
    if slope == 0.0 {
        D::constant((-a2 + 1.0) / a2)
    } else if slope > 0.0 {
        let margin = -a2 + 1.0;
        let shifted = (-vgsteff.scale(a1)).plus(margin - 0.0001);
        let gap = smooth_positive(shifted, D::constant(margin * 0.0004));
        gap / (1.0 - gap)
    } else {
        let shifted = vgsteff.scale(a1).plus(a2 - 0.0001);
        let lambda = smooth_positive(shifted, D::constant(a2 * 0.0004));
        (1.0 - lambda) / lambda
    }
}

/// The saturation voltage `Vdsat` of `b3ld.c` and `b4v5ld.c`, rationalized to
/// `2 · c / (b + √(b² − 2·a·c))` where `b ≥ 0` so `b` and its root never cancel when `a`
/// is small (as with `rds = 0` and `λ → 1`), and continuous through `a = 0` without a
/// special branch.
pub(crate) fn saturation_voltage<R: Real>(
    vgst2vtm: D<R>,
    esat_l: D<R>,
    abulk: D<R>,
    wv_cox_rds: D<R>,
    excess: D<R>,
) -> D<R> {
    let scaled_rds = abulk * wv_cox_rds;
    let leading = abulk * (scaled_rds + excess) * 2.0;
    let linear = vgst2vtm * (excess * 2.0 + 1.0) + abulk * esat_l + (vgst2vtm * scaled_rds) * 3.0;
    let constant = vgst2vtm * (esat_l + (vgst2vtm * wv_cox_rds) * 2.0);
    let root = (linear * linear - leading * constant * 2.0).sqrt();
    if linear.value.value() >= 0.0 {
        (constant * 2.0) / (linear + root)
    } else {
        (linear - root) / leading
    }
}

/// The substrate current's Early voltage `Vascbe` of `b3ld.c` and `b4v5ld.c`.
pub(crate) fn substrate_early_voltage<R: Real>(
    diff_vds: D<R>,
    (pscbe1, pscbe2): (R, R),
    (leff, litl): (R, R),
) -> D<R> {
    if pscbe2.value() > 0.0 {
        if diff_vds.value.value() > pscbe1.value() * litl.value() / EXP_THRESHOLD {
            let t0 = diff_vds.recip().scale(pscbe1 * litl);
            t0.exp().scale(leff / pscbe2)
        } else {
            D::constant(leff / pscbe2 * MAX_EXP)
        }
    } else {
        D::number(MAX_EXP)
    }
}

/// The impact ionization multiplier
/// `(alpha0 / leff + alpha1) · diff_vds · exp(−beta0 / diff_vds)` of `b3ld.c` and
/// `b4v5ld.c`, or `None` when impact ionization is inactive.
pub(crate) fn impact_ionization_factor<R: Real>(
    diff_vds: D<R>,
    (alpha0, alpha1, beta0): (R, R, R),
    leff: R,
) -> Option<D<R>> {
    let tmp = alpha0 + alpha1 * leff;
    if tmp.value() <= 0.0 || beta0.value() <= 0.0 {
        return None;
    }
    let t2 = tmp / leff;
    let factor = if diff_vds.value.value() > beta0.value() / EXP_THRESHOLD {
        let t0 = -diff_vds.recip().scale(beta0);
        (diff_vds * t0.exp()).scale(t2)
    } else {
        diff_vds.scale(t2 * MIN_EXP)
    };
    Some(factor)
}

/// `x0 − ½ · (v + √(v² ± 4δ · x0))`, the smooth minimum of `b3ld.c`'s and `b4v5ld.c`'s
/// effective flat-band voltage, with `v = x0 − x − δ`, the sign of the root's term that
/// of `x0`.
pub(crate) fn flat_band<R: Real>(vfb: R, v3: D<R>) -> D<R> {
    let t0 = if vfb.value() <= 0.0 {
        (v3 * v3 - D::constant(vfb * (4.0 * DELTA))).sqrt()
    } else {
        (v3 * v3 + D::constant(vfb * (4.0 * DELTA))).sqrt()
    };
    (-((v3 + t0) * 0.5)).plus(vfb)
}

/// The depletion charge under the gate, `CoxWL · k1ox · (√(k1ox²/4 + v) − k1ox/2)`,
/// continued linearly below `v = 0`, with `capacitance` the gate's oxide capacitance.
pub(crate) fn depletion<R: Real>(k1ox: R, voltage: D<R>, capacitance: D<R>) -> D<R> {
    let t0 = k1ox * 0.5;
    let t1 = if k1ox.value() == 0.0 {
        D::number(0.0)
    } else if voltage.value.value() < 0.0 {
        voltage.divide(k1ox).plus(t0)
    } else {
        (voltage.plus(t0 * t0)).sqrt()
    };
    capacitance * t1.plus(-t0).scale(k1ox)
}

/// The effective gate overdrive of the capacitance model, shifted by `voffcv` and scaled
/// by `noff`, as `b3ld.c` (`capmod` 1 to 3) and `b4v5ld.c` (`capmod` 2) compute it.
pub(crate) fn cv_overdrive<R: Real>(
    gate_source: D<R>,
    threshold: D<R>,
    swing: D<R>,
    (noff, voffcv, vtm): (R, R, R),
) -> D<R> {
    let scale = swing.scale(noff).scale(vtm);
    let vgst = gate_source - threshold;
    let vgst_nvt = vgst.plus(-voffcv) / scale;
    if vgst_nvt.value.value() > EXP_THRESHOLD {
        vgst.plus(-voffcv)
    } else if vgst_nvt.value.value() < -EXP_THRESHOLD {
        scale * MIN_EXP.ln_1p()
    } else {
        scale * vgst_nvt.exp().ln_1p()
    }
}

/// The accumulation and depletion charge centroid `Tcen` of the charge-thickness model
/// (`b3ld.c`'s `capmod` 3 and `b4v5ld.c`'s `capmod` 2), smoothly bounded above by the
/// Debye length `debye_length`.
pub(crate) fn accumulation_centroid<R: Real>(
    gate_bulk_over_flat: D<R>,
    oxide_thickness: R,
    acde: R,
    debye_length: R,
) -> D<R> {
    let tmp = gate_bulk_over_flat.divide(oxide_thickness * 1.0e8).scale(acde);
    let tcen = if -EXP_THRESHOLD < tmp.value.value() && tmp.value.value() < EXP_THRESHOLD {
        tmp.exp().scale(debye_length)
    } else if tmp.value.value() <= -EXP_THRESHOLD {
        D::constant(debye_length * MIN_EXP)
    } else {
        D::constant(debye_length * MAX_EXP)
    };
    let link = oxide_thickness * 1.0e-3;
    let v3 = (-tcen).plus(debye_length - link);
    let v4 = (v3 * v3).plus(link * debye_length * 4.0).sqrt();
    (-((v3 + v4) * 0.5)).plus(debye_length)
}

/// The inversion charge centroid `Tcen = 1.9 nm / (1 + field^0.7)` of the
/// charge-thickness model (`b3ld.c` and `b4v5ld.c`), at the normalized vertical field
/// `field`.
pub(crate) fn inversion_centroid<R: Real>(field: D<R>) -> D<R> {
    let tmp = (field.ln() * 0.7).exp();
    (tmp + 1.0).recip() * 1.9e-9
}

/// The effective oxide capacitance over the channel's area for a charge centroid
/// `centroid` in the charge-thickness model (`b3ld.c`'s `capmod` 3 and `b4v5ld.c`'s
/// `capmod` 2): `Coxeff · (channel_oxide_cap / electrical_oxide_cap)` where
/// `Ccen = εsi / centroid` and
/// `Coxeff = Ccen · physical_oxide_cap / (Ccen + physical_oxide_cap)`.
pub(crate) fn effective_oxide_capacitance<R: Real>(
    centroid: D<R>,
    physical_oxide_cap: R,
    electrical_oxide_cap: R,
    channel_oxide_cap: R,
) -> D<R> {
    let ccen = centroid.recip() * EPSSI;
    let coxeff = ccen.plus(physical_oxide_cap).recip().scale(physical_oxide_cap) * ccen;
    coxeff.scale(channel_oxide_cap / electrical_oxide_cap)
}

/// The gate overdrive less the surface potential's rise `δφ` in the charge-thickness
/// model (`b3ld.c`'s `capmod` 3 and `b4v5ld.c`'s `capmod` 2), using `ln_1p` and
/// rationalized [`smooth_positive`] to preserve precision in deep subthreshold.
pub(crate) fn surface_potential_overdrive<R: Real>(
    vgsteff: D<R>,
    k1ox: R,
    sqrt_phi: R,
    moin: R,
    vtm: R,
) -> D<R> {
    let (denominator, t0) = if k1ox.value() <= 0.0 {
        (moin * vtm * 0.25, sqrt_phi * 0.5)
    } else {
        (moin * vtm * k1ox * k1ox, k1ox * sqrt_phi)
    };
    let t1 = vgsteff.plus(t0 * 2.0);
    let delta_phi = (t1 * vgsteff).divide(denominator).ln_1p().scale(vtm);
    let shifted = vgsteff - delta_phi - 0.001;
    smooth_positive(shifted, vgsteff * 0.004)
}

/// The source's share of the inversion charge under Ward-Dutton charge partitioning
/// (`xpart`), given `partition`, `(overdrive, scaled_drain, denominator)` where
/// `scaled_drain = Abulk · Vds` and
/// `denominator = 12 · (overdrive − scaled_drain / 2 + 1e-20)`, the oxide capacitance
/// `cox_wl`, and the total inversion charge `inversion`.
pub(crate) fn source_partition_charge<R: Real>(
    partition: Partition,
    (overdrive, scaled_drain, denominator): (D<R>, D<R>, D<R>),
    cox_wl: D<R>,
    inversion: D<R>,
) -> D<R> {
    match partition {
        Partition::None => {
            let quadratic = scaled_drain * scaled_drain * 0.5 / denominator;
            -(cox_wl * (overdrive * 0.5 + scaled_drain * 0.25 - quadratic))
        }
        Partition::Forty => {
            let scaled_denominator = denominator * (1.0 / 12.0);
            let scale = cox_wl * 0.5 / (scaled_denominator * scaled_denominator);
            let polynomial = overdrive
                * (scaled_drain * scaled_drain * (2.0 / 3.0)
                    + overdrive * (overdrive - scaled_drain * (4.0 / 3.0)))
                - scaled_drain * scaled_drain * scaled_drain * (2.0 / 15.0);
            -(scale * polynomial)
        }
        Partition::Half => -(inversion * 0.5),
    }
}

/// Evaluates the charge-thickness inversion and terminal charges
/// `[gate, bulk, drain, inversion]` shared by BSIM3 `capmod` 3 (`b3ld.c`) and BSIM4
/// `capmod` 2 (`b4v5ld.c`).
pub(crate) fn charge_thickness_charges<R: Real>(
    drain_source: D<R>,
    overdrive: D<R>,
    abulk_cv: D<R>,
    cox_wl_cen: D<R>,
    (qac0, qsub0): (D<R>, D<R>),
    partition: Partition,
) -> [D<R>; 4] {
    let vdsat_cv = overdrive / abulk_cv;
    let vdseff_cv = effective_drain(drain_source, vdsat_cv, R::constant(DELTA));
    let t0 = abulk_cv * vdseff_cv;
    let t2 = (overdrive - t0 * 0.5 + 1.0e-20) * 12.0;
    let t3 = t0 / t2;
    let inversion = cox_wl_cen * (overdrive - t0 * (0.5 - t3));
    let bulk = cox_wl_cen * (1.0 - abulk_cv) * (vdseff_cv * 0.5 - t0 * vdseff_cv / t2);
    let source = source_partition_charge(partition, (overdrive, t0, t2), cox_wl_cen, inversion);
    let gate = inversion + qac0 + qsub0 - bulk;
    let bulk = bulk - (qac0 + qsub0);
    let drain = -(gate + bulk + source);
    [gate, bulk, drain, inversion]
}

/// The `DELTA`-smoothed bias-dependent gate-overlap charge of `b3ld.c` (`capmod` 2 and 3)
/// and `b4v5ld.c` (`capmod` 2), at terminal `voltage` with fixed overlap `fixed`,
/// lightly-doped overlap `lightly`, and parameter `kappa`.
pub(crate) fn smooth_overlap<R: Real>(voltage: D<R>, fixed: R, lightly: R, kappa: R) -> D<R> {
    let t0 = voltage + DELTA;
    let t1 = (t0 * t0 + 4.0 * DELTA).sqrt();
    let t2 = (t0 - t1) * 0.5;
    let t4 = (1.0 - t2.divide(kappa) * 4.0).sqrt();
    voltage.scale(fixed + lightly) - (t2 + (t4 - 1.0).scale(kappa * 0.5)).scale(lightly)
}

/// The depletion charge of a three-part junction (bottom, isolation sidewall and gate
/// sidewall) with zero-bias `capacitances`, built-in `potentials` and grading
/// coefficients `gradings` at its `voltage`: each part's power law in reverse bias, and
/// its linear continuation in forward bias, as `b3ld.c` and `b4v5ld.c` compute it.
pub(crate) fn three_part_depletion<R: Real, const N: usize>(
    capacitances: [R; 3],
    potentials: [R; 3],
    gradings: [R; 3],
    voltage: Dual<R, N>,
) -> Dual<R, N> {
    if voltage.value.value() < 0.0 {
        capacitances
            .into_iter()
            .zip(potentials)
            .zip(gradings)
            .fold(
                Dual::number(0.0),
                |sum, ((capacitance, potential), grading)| {
                    if capacitance.value() <= 0.0 {
                        return sum;
                    }
                    let arg = 1.0 - voltage.divide(potential);
                    #[expect(
                        clippy::float_cmp,
                        reason = "b3ld.c and b4v5ld.c take the square root at exactly 0.5"
                    )]
                    let sarg = if grading.value() == 0.5 {
                        arg.sqrt().recip()
                    } else {
                        (arg.ln().scale(-grading)).exp()
                    };
                    sum + (1.0 - arg * sarg).scale(potential * capacitance / (-grading + 1.0))
                },
            )
    } else {
        let [bottom, sidewall, gate] = capacitances;
        let total = bottom + sidewall + gate;
        let slope = capacitances
            .into_iter()
            .zip(potentials)
            .zip(gradings)
            .fold(
                R::constant(0.0),
                |sum, ((capacitance, potential), grading)| {
                    sum + capacitance * grading / potential
                },
            );
        voltage * voltage.scale(slope * 0.5).plus(total)
    }
}

/// The channel-length modulation length of BSIM3's and BSIM4's strong-inversion flicker
/// noise (`b3noi.c` and `b4v5noi.c`).
pub(crate) fn clm_noise_length(vds: f64, vdseff: f64, litl: f64, esat: f64, em: f64) -> f64 {
    if em <= 0.0 {
        0.0
    } else {
        let t0 = ((vds - vdseff) / litl + em) / esat;
        (litl * t0.max(N_MINLOG).ln()).max(0.0)
    }
}
