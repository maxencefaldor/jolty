//! The parameters of a BSIM3 transistor, by the names of the BSIM3v3.3 manual, in SI
//! units, with ngspice's defaults.
//!
//! The defaults are those of ngspice 47's `b3set.c`; a parameter whose default derives
//! from others, as `vth0` from `vfb`, is optional, and its documentation says what it
//! derives from.
//!
//! Part of jolty's BSIM3, derived from BSIM3v3.3.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2004 Regents of the University of California. All rights reserved. Used
//! under the BSIM terms of use, which the repository's `NOTICE` reproduces.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{Charges, Mobility, NoiseModel, by_channel, fringe};
use crate::kernel::jfet::Channel;
use crate::{Lift, REFERENCE_TEMPERATURE, Real};

/// The parameters of a [`Bsim3`](super::Bsim3): the instance's geometry, then the model's
/// card. Each parameter of the card keeps the name the BSIM3v3.3 manual gives it, and
/// each one that bins has a length, a width and an area dependence, `l`, `w` and `p`
/// before its name, in SI units: a dependence on the reciprocal of the effective length
/// in meters, not micrometers.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields, bound = "")]
pub struct Bsim3Parameters<R: Real = f64> {
    /// The channel; default `n`.
    pub channel: Channel,
    /// The channel's drawn width, `w`, in meters; default 5 µm.
    pub width: R,
    /// The channel's drawn length, `l`, in meters; default 5 µm.
    pub length: R,
    /// The number of identical transistors in parallel, `m`; default 1.
    pub multiplier: R,
    /// The drain diffusion's area, `ad`, in square meters; default 0.
    pub drain_area: R,
    /// The source diffusion's area, `as`, in square meters; default 0.
    pub source_area: R,
    /// The drain diffusion's perimeter, `pd`, in meters; default 0.
    pub drain_perimeter: R,
    /// The source diffusion's perimeter, `ps`, in meters; default 0.
    pub source_perimeter: R,
    /// The drain diffusion's length in squares, `nrd`; default 1.
    pub drain_squares: R,
    /// The source diffusion's length in squares, `nrs`; default 1.
    pub source_squares: R,
    /// A shift of the zero-bias threshold, `delvto`, in volts; default 0.
    pub threshold_adjustment: R,
    /// A factor of the low-field mobility, `mulu0`; default 1.
    pub mobility_multiplier: R,
    /// The mobility model, `mobmod`, 1, 2 or 3; default 1.
    pub mobility_model: Mobility,
    /// The charge model, `capmod`, 0 to 3; default 3.
    pub capacitance_model: Charges,
    /// The noise model, `noimod`, 1 to 6; default 1.
    pub noise_model: NoiseModel,
    /// Whether the size-dependent parameters are checked and clamped where they lie out
    /// of range, `paramchk = 1`; default false.
    pub check_parameters: bool,
    /// The share of the channel's charge the drain takes in saturation, `xpart`: 0.4 below
    /// 0.5, 0.5 at 0.5 and none above it, and no channel charge at all below 0; default 0.
    pub charge_partition: f64,
    /// The temperature the parameters were measured at, `tnom`, in kelvins; default 27 °C.
    pub nominal_temperature: R,
    /// The gate oxide's thickness, `tox`, in meters; default 1.5e-8.
    pub tox: R,
    /// The oxide thickness the parameters were extracted at, which scales `k1` and
    /// `k2`, `toxm`, in meters; default `tox`.
    pub toxm: Option<R>,
    /// The source and drain junctions' depth, `xj`, in meters; default 1.5e-7.
    pub xj: R,
    /// The length dependence of `xj`, `lxj`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `xj` times meters; default 0.
    pub lxj: R,
    /// The width dependence of `xj`, `wxj`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `xj` times meters; default 0.
    pub wxj: R,
    /// The area dependence of `xj`, `pxj`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `xj` times square
    /// meters; default 0.
    pub pxj: R,
    /// The channel's doping concentration near the surface, `nch`, per cubic meter;
    /// default 1.7e23 per cubic meter, or derived from `gamma1` where it alone is
    /// given.
    pub nch: Option<R>,
    /// The length dependence of `nch`, `lnch`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `nch` times meters; default 0.
    pub lnch: R,
    /// The width dependence of `nch`, `wnch`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `nch` times meters; default 0.
    pub wnch: R,
    /// The area dependence of `nch`, `pnch`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `nch` times square
    /// meters; default 0.
    pub pnch: R,
    /// The substrate's doping concentration, `nsub`, per cubic meter; default 6e22.
    pub nsub: R,
    /// The length dependence of `nsub`, `lnsub`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `nsub` times meters; default 0.
    pub lnsub: R,
    /// The width dependence of `nsub`, `wnsub`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `nsub` times meters; default 0.
    pub wnsub: R,
    /// The area dependence of `nsub`, `pnsub`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `nsub` times square
    /// meters; default 0.
    pub pnsub: R,
    /// The poly gate's doping concentration, zero for no depletion of the gate,
    /// `ngate`, per cubic meter; default 0.
    pub ngate: R,
    /// The length dependence of `ngate`, `lngate`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `ngate` times meters; default 0.
    pub lngate: R,
    /// The width dependence of `ngate`, `wngate`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `ngate` times meters; default 0.
    pub wngate: R,
    /// The area dependence of `ngate`, `pngate`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `ngate` times
    /// square meters; default 0.
    pub pngate: R,
    /// The body effect near the surface, `gamma1`, in square-root volts; default
    /// derived from `nch`.
    pub gamma1: Option<R>,
    /// The length dependence of `gamma1`, `lgamma1`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `gamma1` times meters; default 0.
    pub lgamma1: R,
    /// The width dependence of `gamma1`, `wgamma1`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `gamma1` times meters; default 0.
    pub wgamma1: R,
    /// The area dependence of `gamma1`, `pgamma1`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `gamma1` times
    /// square meters; default 0.
    pub pgamma1: R,
    /// The body effect in the bulk, `gamma2`, in square-root volts; default derived
    /// from `nsub`.
    pub gamma2: Option<R>,
    /// The length dependence of `gamma2`, `lgamma2`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `gamma2` times meters; default 0.
    pub lgamma2: R,
    /// The width dependence of `gamma2`, `wgamma2`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `gamma2` times meters; default 0.
    pub wgamma2: R,
    /// The area dependence of `gamma2`, `pgamma2`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `gamma2` times
    /// square meters; default 0.
    pub pgamma2: R,
    /// The bulk-source voltage at which the depletion reaches `xt`, `vbx`, in volts;
    /// default derived from `nch` and `xt`.
    pub vbx: Option<R>,
    /// The length dependence of `vbx`, `lvbx`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `vbx` times meters; default 0.
    pub lvbx: R,
    /// The width dependence of `vbx`, `wvbx`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `vbx` times meters; default 0.
    pub wvbx: R,
    /// The area dependence of `vbx`, `pvbx`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `vbx` times square
    /// meters; default 0.
    pub pvbx: R,
    /// The most negative bulk-source voltage the threshold's body effect reads, `vbm`,
    /// in volts; default -3.
    pub vbm: R,
    /// The length dependence of `vbm`, `lvbm`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `vbm` times meters; default 0.
    pub lvbm: R,
    /// The width dependence of `vbm`, `wvbm`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `vbm` times meters; default 0.
    pub wvbm: R,
    /// The area dependence of `vbm`, `pvbm`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `vbm` times square
    /// meters; default 0.
    pub pvbm: R,
    /// The depth of the doping near the surface, `xt`, in meters; default 1.55e-7.
    pub xt: R,
    /// The length dependence of `xt`, `lxt`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `xt` times meters; default 0.
    pub lxt: R,
    /// The width dependence of `xt`, `wxt`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `xt` times meters; default 0.
    pub wxt: R,
    /// The area dependence of `xt`, `pxt`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `xt` times square
    /// meters; default 0.
    pub pxt: R,
    /// The zero-bias threshold voltage of a long, wide channel, negative for a
    /// p-channel, `vth0`, in volts; default derived from `vfb`, `phi` and `k1`.
    pub vth0: Option<R>,
    /// The length dependence of `vth0`, `lvth0`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `vth0` times meters; default 0.
    pub lvth0: R,
    /// The width dependence of `vth0`, `wvth0`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `vth0` times meters; default 0.
    pub wvth0: R,
    /// The area dependence of `vth0`, `pvth0`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `vth0` times square
    /// meters; default 0.
    pub pvth0: R,
    /// The flat-band voltage, `vfb`, in volts; default derived from `vth0`, or -1 V.
    pub vfb: Option<R>,
    /// The length dependence of `vfb`, `lvfb`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `vfb` times meters; default 0.
    pub lvfb: R,
    /// The width dependence of `vfb`, `wvfb`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `vfb` times meters; default 0.
    pub wvfb: R,
    /// The area dependence of `vfb`, `pvfb`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `vfb` times square
    /// meters; default 0.
    pub pvfb: R,
    /// The first-order body effect, `k1`, in square-root volts; default derived from
    /// `gamma1` and `gamma2`, or 0.53 √V where only `k2` is given.
    pub k1: Option<R>,
    /// The length dependence of `k1`, `lk1`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `k1` times meters; default 0.
    pub lk1: R,
    /// The width dependence of `k1`, `wk1`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `k1` times meters; default 0.
    pub wk1: R,
    /// The area dependence of `k1`, `pk1`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `k1` times square
    /// meters; default 0.
    pub pk1: R,
    /// The second-order body effect, `k2`; default derived from `gamma1` and `gamma2`,
    /// or -0.0186 where only `k1` is given.
    pub k2: Option<R>,
    /// The length dependence of `k2`, `lk2`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `k2` times meters; default 0.
    pub lk2: R,
    /// The width dependence of `k2`, `wk2`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `k2` times meters; default 0.
    pub wk2: R,
    /// The area dependence of `k2`, `pk2`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `k2` times square
    /// meters; default 0.
    pub pk2: R,
    /// The narrow-width coefficient, `k3`; default 80.
    pub k3: R,
    /// The length dependence of `k3`, `lk3`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `k3` times meters; default 0.
    pub lk3: R,
    /// The width dependence of `k3`, `wk3`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `k3` times meters; default 0.
    pub wk3: R,
    /// The area dependence of `k3`, `pk3`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `k3` times square
    /// meters; default 0.
    pub pk3: R,
    /// The body effect on `k3`, `k3b`, per volt; default 0.
    pub k3b: R,
    /// The length dependence of `k3b`, `lk3b`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `k3b` times meters; default 0.
    pub lk3b: R,
    /// The width dependence of `k3b`, `wk3b`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `k3b` times meters; default 0.
    pub wk3b: R,
    /// The area dependence of `k3b`, `pk3b`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `k3b` times square
    /// meters; default 0.
    pub pk3b: R,
    /// The width offset of the narrow-width effect, `w0`, in meters; default 2.5e-6.
    pub w0: R,
    /// The length dependence of `w0`, `lw0`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `w0` times meters; default 0.
    pub lw0: R,
    /// The width dependence of `w0`, `ww0`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `w0` times meters; default 0.
    pub ww0: R,
    /// The area dependence of `w0`, `pw0`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `w0` times square
    /// meters; default 0.
    pub pw0: R,
    /// The lateral non-uniform doping along the channel, `nlx`, in meters; default
    /// 1.74e-7.
    pub nlx: R,
    /// The length dependence of `nlx`, `lnlx`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `nlx` times meters; default 0.
    pub lnlx: R,
    /// The width dependence of `nlx`, `wnlx`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `nlx` times meters; default 0.
    pub wnlx: R,
    /// The area dependence of `nlx`, `pnlx`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `nlx` times square
    /// meters; default 0.
    pub pnlx: R,
    /// The first coefficient of the short-channel effect, `dvt0`; default 2.2.
    pub dvt0: R,
    /// The length dependence of `dvt0`, `ldvt0`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `dvt0` times meters; default 0.
    pub ldvt0: R,
    /// The width dependence of `dvt0`, `wdvt0`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `dvt0` times meters; default 0.
    pub wdvt0: R,
    /// The area dependence of `dvt0`, `pdvt0`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `dvt0` times square
    /// meters; default 0.
    pub pdvt0: R,
    /// The second coefficient of the short-channel effect, `dvt1`; default 0.53.
    pub dvt1: R,
    /// The length dependence of `dvt1`, `ldvt1`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `dvt1` times meters; default 0.
    pub ldvt1: R,
    /// The width dependence of `dvt1`, `wdvt1`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `dvt1` times meters; default 0.
    pub wdvt1: R,
    /// The area dependence of `dvt1`, `pdvt1`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `dvt1` times square
    /// meters; default 0.
    pub pdvt1: R,
    /// The body effect on the short-channel effect, `dvt2`, per volt; default -0.032.
    pub dvt2: R,
    /// The length dependence of `dvt2`, `ldvt2`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `dvt2` times meters; default 0.
    pub ldvt2: R,
    /// The width dependence of `dvt2`, `wdvt2`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `dvt2` times meters; default 0.
    pub wdvt2: R,
    /// The area dependence of `dvt2`, `pdvt2`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `dvt2` times square
    /// meters; default 0.
    pub pdvt2: R,
    /// The first coefficient of the narrow-width effect on a short channel, `dvt0w`;
    /// default 0.
    pub dvt0w: R,
    /// The length dependence of `dvt0w`, `ldvt0w`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `dvt0w` times meters; default 0.
    pub ldvt0w: R,
    /// The width dependence of `dvt0w`, `wdvt0w`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `dvt0w` times meters; default 0.
    pub wdvt0w: R,
    /// The area dependence of `dvt0w`, `pdvt0w`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `dvt0w` times
    /// square meters; default 0.
    pub pdvt0w: R,
    /// The second coefficient of the narrow-width effect on a short channel, `dvt1w`,
    /// per meter; default 5300000.
    pub dvt1w: R,
    /// The length dependence of `dvt1w`, `ldvt1w`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `dvt1w` times meters; default 0.
    pub ldvt1w: R,
    /// The width dependence of `dvt1w`, `wdvt1w`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `dvt1w` times meters; default 0.
    pub wdvt1w: R,
    /// The area dependence of `dvt1w`, `pdvt1w`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `dvt1w` times
    /// square meters; default 0.
    pub pdvt1w: R,
    /// The body effect on the narrow-width effect on a short channel, `dvt2w`, per
    /// volt; default -0.032.
    pub dvt2w: R,
    /// The length dependence of `dvt2w`, `ldvt2w`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `dvt2w` times meters; default 0.
    pub ldvt2w: R,
    /// The width dependence of `dvt2w`, `wdvt2w`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `dvt2w` times meters; default 0.
    pub wdvt2w: R,
    /// The area dependence of `dvt2w`, `pdvt2w`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `dvt2w` times
    /// square meters; default 0.
    pub pdvt2w: R,
    /// The drain-induced barrier lowering's coefficient below the threshold, `dsub`;
    /// default `drout`.
    pub dsub: Option<R>,
    /// The length dependence of `dsub`, `ldsub`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `dsub` times meters; default 0.
    pub ldsub: R,
    /// The width dependence of `dsub`, `wdsub`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `dsub` times meters; default 0.
    pub wdsub: R,
    /// The area dependence of `dsub`, `pdsub`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `dsub` times square
    /// meters; default 0.
    pub pdsub: R,
    /// The drain-induced barrier lowering below the threshold, `eta0`; default 0.08.
    pub eta0: R,
    /// The length dependence of `eta0`, `leta0`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `eta0` times meters; default 0.
    pub leta0: R,
    /// The width dependence of `eta0`, `weta0`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `eta0` times meters; default 0.
    pub weta0: R,
    /// The area dependence of `eta0`, `peta0`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `eta0` times square
    /// meters; default 0.
    pub peta0: R,
    /// The body effect on the drain-induced barrier lowering below the threshold,
    /// `etab`, per volt; default -0.07.
    pub etab: R,
    /// The length dependence of `etab`, `letab`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `etab` times meters; default 0.
    pub letab: R,
    /// The width dependence of `etab`, `wetab`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `etab` times meters; default 0.
    pub wetab: R,
    /// The area dependence of `etab`, `petab`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `etab` times square
    /// meters; default 0.
    pub petab: R,
    /// The low-field mobility at the nominal temperature, `u0`, in square meters per
    /// volt-second; default 0.067 m²/(V·s) for an n-channel, 0.025 m²/(V·s) for a
    /// p-channel.
    pub u0: Option<R>,
    /// The length dependence of `u0`, `lu0`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `u0` times meters; default 0.
    pub lu0: R,
    /// The width dependence of `u0`, `wu0`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `u0` times meters; default 0.
    pub wu0: R,
    /// The area dependence of `u0`, `pu0`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `u0` times square
    /// meters; default 0.
    pub pu0: R,
    /// The first-order degradation of the mobility by the gate field, `ua`, in meters
    /// per volt; default 2.25e-9.
    pub ua: R,
    /// The length dependence of `ua`, `lua`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `ua` times meters; default 0.
    pub lua: R,
    /// The width dependence of `ua`, `wua`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `ua` times meters; default 0.
    pub wua: R,
    /// The area dependence of `ua`, `pua`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `ua` times square
    /// meters; default 0.
    pub pua: R,
    /// The second-order degradation of the mobility by the gate field, `ub`, in square
    /// meters per square volt; default 5.87e-19.
    pub ub: R,
    /// The length dependence of `ub`, `lub`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `ub` times meters; default 0.
    pub lub: R,
    /// The width dependence of `ub`, `wub`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `ub` times meters; default 0.
    pub wub: R,
    /// The area dependence of `ub`, `pub`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `ub` times square
    /// meters; default 0.
    pub r#pub: R,
    /// The body effect on the mobility's degradation, `uc`, in meters per square volt,
    /// or per volt at `mobmod` 3; default -4.65e-11 m/V², or -0.0465 per volt at
    /// `mobmod` 3.
    pub uc: Option<R>,
    /// The length dependence of `uc`, `luc`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `uc` times meters; default 0.
    pub luc: R,
    /// The width dependence of `uc`, `wuc`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `uc` times meters; default 0.
    pub wuc: R,
    /// The area dependence of `uc`, `puc`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `uc` times square
    /// meters; default 0.
    pub puc: R,
    /// The carriers' saturation velocity at the nominal temperature, `vsat`, in meters
    /// per second; default 80000.
    pub vsat: R,
    /// The length dependence of `vsat`, `lvsat`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `vsat` times meters; default 0.
    pub lvsat: R,
    /// The width dependence of `vsat`, `wvsat`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `vsat` times meters; default 0.
    pub wvsat: R,
    /// The area dependence of `vsat`, `pvsat`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `vsat` times square
    /// meters; default 0.
    pub pvsat: R,
    /// The bulk charge effect of a short channel, `a0`; default 1.
    pub a0: R,
    /// The length dependence of `a0`, `la0`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `a0` times meters; default 0.
    pub la0: R,
    /// The width dependence of `a0`, `wa0`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `a0` times meters; default 0.
    pub wa0: R,
    /// The area dependence of `a0`, `pa0`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `a0` times square
    /// meters; default 0.
    pub pa0: R,
    /// The gate bias's effect on the bulk charge, `ags`, per volt; default 0.
    pub ags: R,
    /// The length dependence of `ags`, `lags`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `ags` times meters; default 0.
    pub lags: R,
    /// The width dependence of `ags`, `wags`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `ags` times meters; default 0.
    pub wags: R,
    /// The area dependence of `ags`, `pags`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `ags` times square
    /// meters; default 0.
    pub pags: R,
    /// The first coefficient of the non-saturation effect, `a1`, per volt; default 0.
    pub a1: R,
    /// The length dependence of `a1`, `la1`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `a1` times meters; default 0.
    pub la1: R,
    /// The width dependence of `a1`, `wa1`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `a1` times meters; default 0.
    pub wa1: R,
    /// The area dependence of `a1`, `pa1`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `a1` times square
    /// meters; default 0.
    pub pa1: R,
    /// The second coefficient of the non-saturation effect, `a2`; default 1.
    pub a2: R,
    /// The length dependence of `a2`, `la2`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `a2` times meters; default 0.
    pub la2: R,
    /// The width dependence of `a2`, `wa2`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `a2` times meters; default 0.
    pub wa2: R,
    /// The area dependence of `a2`, `pa2`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `a2` times square
    /// meters; default 0.
    pub pa2: R,
    /// The narrow-width effect on the bulk charge, `b0`, in meters; default 0.
    pub b0: R,
    /// The length dependence of `b0`, `lb0`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `b0` times meters; default 0.
    pub lb0: R,
    /// The width dependence of `b0`, `wb0`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `b0` times meters; default 0.
    pub wb0: R,
    /// The area dependence of `b0`, `pb0`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `b0` times square
    /// meters; default 0.
    pub pb0: R,
    /// The width offset of the narrow-width effect on the bulk charge, `b1`, in meters;
    /// default 0.
    pub b1: R,
    /// The length dependence of `b1`, `lb1`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `b1` times meters; default 0.
    pub lb1: R,
    /// The width dependence of `b1`, `wb1`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `b1` times meters; default 0.
    pub wb1: R,
    /// The area dependence of `b1`, `pb1`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `b1` times square
    /// meters; default 0.
    pub pb1: R,
    /// The body effect on the bulk charge, `keta`, per volt; default -0.047.
    pub keta: R,
    /// The length dependence of `keta`, `lketa`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `keta` times meters; default 0.
    pub lketa: R,
    /// The width dependence of `keta`, `wketa`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `keta` times meters; default 0.
    pub wketa: R,
    /// The area dependence of `keta`, `pketa`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `keta` times square
    /// meters; default 0.
    pub pketa: R,
    /// The offset of the threshold below it, in the drain current's smoothing, `voff`,
    /// in volts; default -0.08.
    pub voff: R,
    /// The length dependence of `voff`, `lvoff`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `voff` times meters; default 0.
    pub lvoff: R,
    /// The width dependence of `voff`, `wvoff`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `voff` times meters; default 0.
    pub wvoff: R,
    /// The area dependence of `voff`, `pvoff`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `voff` times square
    /// meters; default 0.
    pub pvoff: R,
    /// The subthreshold swing's coefficient, `nfactor`; default 1.
    pub nfactor: R,
    /// The length dependence of `nfactor`, `lnfactor`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `nfactor` times meters;
    /// default 0.
    pub lnfactor: R,
    /// The width dependence of `nfactor`, `wnfactor`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `nfactor` times meters; default 0.
    pub wnfactor: R,
    /// The area dependence of `nfactor`, `pnfactor`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `nfactor`
    /// times square meters; default 0.
    pub pnfactor: R,
    /// The coupling capacitance between the channel and the source and drain, `cdsc`,
    /// in farads per square meter; default 0.00024.
    pub cdsc: R,
    /// The length dependence of `cdsc`, `lcdsc`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `cdsc` times meters; default 0.
    pub lcdsc: R,
    /// The width dependence of `cdsc`, `wcdsc`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `cdsc` times meters; default 0.
    pub wcdsc: R,
    /// The area dependence of `cdsc`, `pcdsc`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `cdsc` times square
    /// meters; default 0.
    pub pcdsc: R,
    /// The body bias's effect on `cdsc`, `cdscb`, in farads per volt-square-meter;
    /// default 0.
    pub cdscb: R,
    /// The length dependence of `cdscb`, `lcdscb`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `cdscb` times meters; default 0.
    pub lcdscb: R,
    /// The width dependence of `cdscb`, `wcdscb`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `cdscb` times meters; default 0.
    pub wcdscb: R,
    /// The area dependence of `cdscb`, `pcdscb`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `cdscb` times
    /// square meters; default 0.
    pub pcdscb: R,
    /// The drain bias's effect on `cdsc`, `cdscd`, in farads per volt-square-meter;
    /// default 0.
    pub cdscd: R,
    /// The length dependence of `cdscd`, `lcdscd`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `cdscd` times meters; default 0.
    pub lcdscd: R,
    /// The width dependence of `cdscd`, `wcdscd`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `cdscd` times meters; default 0.
    pub wcdscd: R,
    /// The area dependence of `cdscd`, `pcdscd`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `cdscd` times
    /// square meters; default 0.
    pub pcdscd: R,
    /// The interface traps' capacitance, `cit`, in farads per square meter; default 0.
    pub cit: R,
    /// The length dependence of `cit`, `lcit`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `cit` times meters; default 0.
    pub lcit: R,
    /// The width dependence of `cit`, `wcit`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `cit` times meters; default 0.
    pub wcit: R,
    /// The area dependence of `cit`, `pcit`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `cit` times square
    /// meters; default 0.
    pub pcit: R,
    /// The channel-length modulation's coefficient, `pclm`; default 1.3.
    pub pclm: R,
    /// The length dependence of `pclm`, `lpclm`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `pclm` times meters; default 0.
    pub lpclm: R,
    /// The width dependence of `pclm`, `wpclm`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `pclm` times meters; default 0.
    pub wpclm: R,
    /// The area dependence of `pclm`, `ppclm`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `pclm` times square
    /// meters; default 0.
    pub ppclm: R,
    /// The first coefficient of the drain-induced barrier lowering on the output
    /// resistance, `pdiblc1`; default 0.39.
    pub pdiblc1: R,
    /// The length dependence of `pdiblc1`, `lpdiblc1`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `pdiblc1` times meters;
    /// default 0.
    pub lpdiblc1: R,
    /// The width dependence of `pdiblc1`, `wpdiblc1`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `pdiblc1` times meters; default 0.
    pub wpdiblc1: R,
    /// The area dependence of `pdiblc1`, `ppdiblc1`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `pdiblc1`
    /// times square meters; default 0.
    pub ppdiblc1: R,
    /// The second coefficient of the drain-induced barrier lowering on the output
    /// resistance, `pdiblc2`; default 0.0086.
    pub pdiblc2: R,
    /// The length dependence of `pdiblc2`, `lpdiblc2`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `pdiblc2` times meters;
    /// default 0.
    pub lpdiblc2: R,
    /// The width dependence of `pdiblc2`, `wpdiblc2`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `pdiblc2` times meters; default 0.
    pub wpdiblc2: R,
    /// The area dependence of `pdiblc2`, `ppdiblc2`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `pdiblc2`
    /// times square meters; default 0.
    pub ppdiblc2: R,
    /// The body effect on the drain-induced barrier lowering on the output resistance,
    /// `pdiblcb`, per volt; default 0.
    pub pdiblcb: R,
    /// The length dependence of `pdiblcb`, `lpdiblcb`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `pdiblcb` times meters;
    /// default 0.
    pub lpdiblcb: R,
    /// The width dependence of `pdiblcb`, `wpdiblcb`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `pdiblcb` times meters; default 0.
    pub wpdiblcb: R,
    /// The area dependence of `pdiblcb`, `ppdiblcb`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `pdiblcb`
    /// times square meters; default 0.
    pub ppdiblcb: R,
    /// The length dependence of the drain-induced barrier lowering on the output
    /// resistance, `drout`; default 0.56.
    pub drout: R,
    /// The length dependence of `drout`, `ldrout`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `drout` times meters; default 0.
    pub ldrout: R,
    /// The width dependence of `drout`, `wdrout`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `drout` times meters; default 0.
    pub wdrout: R,
    /// The area dependence of `drout`, `pdrout`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `drout` times
    /// square meters; default 0.
    pub pdrout: R,
    /// The first coefficient of the substrate current's body effect on the output
    /// resistance, `pscbe1`, in volts per meter; default 424000000.
    pub pscbe1: R,
    /// The length dependence of `pscbe1`, `lpscbe1`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `pscbe1` times meters; default 0.
    pub lpscbe1: R,
    /// The width dependence of `pscbe1`, `wpscbe1`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `pscbe1` times meters; default 0.
    pub wpscbe1: R,
    /// The area dependence of `pscbe1`, `ppscbe1`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `pscbe1` times
    /// square meters; default 0.
    pub ppscbe1: R,
    /// The second coefficient of the substrate current's body effect on the output
    /// resistance, `pscbe2`, in meters per volt; default 1e-5.
    pub pscbe2: R,
    /// The length dependence of `pscbe2`, `lpscbe2`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `pscbe2` times meters; default 0.
    pub lpscbe2: R,
    /// The width dependence of `pscbe2`, `wpscbe2`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `pscbe2` times meters; default 0.
    pub wpscbe2: R,
    /// The area dependence of `pscbe2`, `ppscbe2`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `pscbe2` times
    /// square meters; default 0.
    pub ppscbe2: R,
    /// The gate bias's effect on the Early voltage, `pvag`; default 0.
    pub pvag: R,
    /// The length dependence of `pvag`, `lpvag`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `pvag` times meters; default 0.
    pub lpvag: R,
    /// The width dependence of `pvag`, `wpvag`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `pvag` times meters; default 0.
    pub wpvag: R,
    /// The area dependence of `pvag`, `ppvag`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `pvag` times square
    /// meters; default 0.
    pub ppvag: R,
    /// The smoothing of the effective drain-source voltage at saturation, `delta`, in
    /// volts; default 0.01.
    pub delta: R,
    /// The length dependence of `delta`, `ldelta`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `delta` times meters; default 0.
    pub ldelta: R,
    /// The width dependence of `delta`, `wdelta`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `delta` times meters; default 0.
    pub wdelta: R,
    /// The area dependence of `delta`, `pdelta`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `delta` times
    /// square meters; default 0.
    pub pdelta: R,
    /// The source and drain resistance of a channel 1 µm wide, which the drain current
    /// folds in, `RDSW` in ohms times micrometers to the power `wr`, `rdsw`, in ohms;
    /// default 0.
    pub rdsw: R,
    /// The length dependence of `rdsw`, `lrdsw`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `rdsw` times meters; default 0.
    pub lrdsw: R,
    /// The width dependence of `rdsw`, `wrdsw`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `rdsw` times meters; default 0.
    pub wrdsw: R,
    /// The area dependence of `rdsw`, `prdsw`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `rdsw` times square
    /// meters; default 0.
    pub prdsw: R,
    /// The gate bias's effect on `rdsw`, `prwg`, per volt; default 0.
    pub prwg: R,
    /// The length dependence of `prwg`, `lprwg`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `prwg` times meters; default 0.
    pub lprwg: R,
    /// The width dependence of `prwg`, `wprwg`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `prwg` times meters; default 0.
    pub wprwg: R,
    /// The area dependence of `prwg`, `pprwg`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `prwg` times square
    /// meters; default 0.
    pub pprwg: R,
    /// The body effect on `rdsw`, `prwb`, per square-root volt; default 0.
    pub prwb: R,
    /// The length dependence of `prwb`, `lprwb`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `prwb` times meters; default 0.
    pub lprwb: R,
    /// The width dependence of `prwb`, `wprwb`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `prwb` times meters; default 0.
    pub wprwb: R,
    /// The area dependence of `prwb`, `pprwb`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `prwb` times square
    /// meters; default 0.
    pub pprwb: R,
    /// The width exponent of `rdsw`, `wr`; default 1.
    pub wr: R,
    /// The length dependence of `wr`, `lwr`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `wr` times meters; default 0.
    pub lwr: R,
    /// The width dependence of `wr`, `wwr`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `wr` times meters; default 0.
    pub wwr: R,
    /// The area dependence of `wr`, `pwr`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `wr` times square
    /// meters; default 0.
    pub pwr: R,
    /// The gate bias's narrowing of the channel, `dwg`, in meters per volt; default 0.
    pub dwg: R,
    /// The length dependence of `dwg`, `ldwg`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `dwg` times meters; default 0.
    pub ldwg: R,
    /// The width dependence of `dwg`, `wdwg`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `dwg` times meters; default 0.
    pub wdwg: R,
    /// The area dependence of `dwg`, `pdwg`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `dwg` times square
    /// meters; default 0.
    pub pdwg: R,
    /// The body bias's narrowing of the channel, `dwb`, in meters per square-root volt;
    /// default 0.
    pub dwb: R,
    /// The length dependence of `dwb`, `ldwb`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `dwb` times meters; default 0.
    pub ldwb: R,
    /// The width dependence of `dwb`, `wdwb`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `dwb` times meters; default 0.
    pub wdwb: R,
    /// The area dependence of `dwb`, `pdwb`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `dwb` times square
    /// meters; default 0.
    pub pdwb: R,
    /// The first coefficient of the impact ionization's substrate current, `alpha0`, in
    /// meters per volt; default 0.
    pub alpha0: R,
    /// The length dependence of `alpha0`, `lalpha0`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `alpha0` times meters; default 0.
    pub lalpha0: R,
    /// The width dependence of `alpha0`, `walpha0`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `alpha0` times meters; default 0.
    pub walpha0: R,
    /// The area dependence of `alpha0`, `palpha0`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `alpha0` times
    /// square meters; default 0.
    pub palpha0: R,
    /// The length scaling of the impact ionization's substrate current, `alpha1`, per
    /// volt; default 0.
    pub alpha1: R,
    /// The length dependence of `alpha1`, `lalpha1`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `alpha1` times meters; default 0.
    pub lalpha1: R,
    /// The width dependence of `alpha1`, `walpha1`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `alpha1` times meters; default 0.
    pub walpha1: R,
    /// The area dependence of `alpha1`, `palpha1`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `alpha1` times
    /// square meters; default 0.
    pub palpha1: R,
    /// The second coefficient of the impact ionization's substrate current, `beta0`, in
    /// volts; default 30.
    pub beta0: R,
    /// The length dependence of `beta0`, `lbeta0`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `beta0` times meters; default 0.
    pub lbeta0: R,
    /// The width dependence of `beta0`, `wbeta0`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `beta0` times meters; default 0.
    pub wbeta0: R,
    /// The area dependence of `beta0`, `pbeta0`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `beta0` times
    /// square meters; default 0.
    pub pbeta0: R,
    /// The threshold's temperature coefficient, `kt1`, in volts; default -0.11.
    pub kt1: R,
    /// The length dependence of `kt1`, `lkt1`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `kt1` times meters; default 0.
    pub lkt1: R,
    /// The width dependence of `kt1`, `wkt1`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `kt1` times meters; default 0.
    pub wkt1: R,
    /// The area dependence of `kt1`, `pkt1`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `kt1` times square
    /// meters; default 0.
    pub pkt1: R,
    /// The channel length's effect on `kt1`, `kt1l`, in volt-meters; default 0.
    pub kt1l: R,
    /// The length dependence of `kt1l`, `lkt1l`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `kt1l` times meters; default 0.
    pub lkt1l: R,
    /// The width dependence of `kt1l`, `wkt1l`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `kt1l` times meters; default 0.
    pub wkt1l: R,
    /// The area dependence of `kt1l`, `pkt1l`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `kt1l` times square
    /// meters; default 0.
    pub pkt1l: R,
    /// The body effect on `kt1`, `kt2`; default 0.022.
    pub kt2: R,
    /// The length dependence of `kt2`, `lkt2`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `kt2` times meters; default 0.
    pub lkt2: R,
    /// The width dependence of `kt2`, `wkt2`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `kt2` times meters; default 0.
    pub wkt2: R,
    /// The area dependence of `kt2`, `pkt2`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `kt2` times square
    /// meters; default 0.
    pub pkt2: R,
    /// The mobility's temperature exponent, `ute`; default -1.5.
    pub ute: R,
    /// The length dependence of `ute`, `lute`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `ute` times meters; default 0.
    pub lute: R,
    /// The width dependence of `ute`, `wute`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `ute` times meters; default 0.
    pub wute: R,
    /// The area dependence of `ute`, `pute`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `ute` times square
    /// meters; default 0.
    pub pute: R,
    /// The temperature coefficient of `ua`, `ua1`, in meters per volt; default 4.31e-9.
    pub ua1: R,
    /// The length dependence of `ua1`, `lua1`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `ua1` times meters; default 0.
    pub lua1: R,
    /// The width dependence of `ua1`, `wua1`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `ua1` times meters; default 0.
    pub wua1: R,
    /// The area dependence of `ua1`, `pua1`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `ua1` times square
    /// meters; default 0.
    pub pua1: R,
    /// The temperature coefficient of `ub`, `ub1`, in square meters per square volt;
    /// default -7.61e-18.
    pub ub1: R,
    /// The length dependence of `ub1`, `lub1`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `ub1` times meters; default 0.
    pub lub1: R,
    /// The width dependence of `ub1`, `wub1`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `ub1` times meters; default 0.
    pub wub1: R,
    /// The area dependence of `ub1`, `pub1`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `ub1` times square
    /// meters; default 0.
    pub pub1: R,
    /// The temperature coefficient of `uc`, `uc1`, in meters per square volt, or per
    /// volt at `mobmod` 3; default -5.6e-11 m/V², or -0.056 per volt at `mobmod` 3.
    pub uc1: Option<R>,
    /// The length dependence of `uc1`, `luc1`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `uc1` times meters; default 0.
    pub luc1: R,
    /// The width dependence of `uc1`, `wuc1`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `uc1` times meters; default 0.
    pub wuc1: R,
    /// The area dependence of `uc1`, `puc1`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `uc1` times square
    /// meters; default 0.
    pub puc1: R,
    /// The temperature coefficient of `vsat`, `at`, in meters per second; default
    /// 33000.
    pub at: R,
    /// The length dependence of `at`, `lat`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `at` times meters; default 0.
    pub lat: R,
    /// The width dependence of `at`, `wat`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `at` times meters; default 0.
    pub wat: R,
    /// The area dependence of `at`, `pat`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `at` times square
    /// meters; default 0.
    pub pat: R,
    /// The temperature coefficient of `rdsw`, as `rdsw` is, `prt`, in ohms; default 0.
    pub prt: R,
    /// The length dependence of `prt`, `lprt`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `prt` times meters; default 0.
    pub lprt: R,
    /// The width dependence of `prt`, `wprt`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `prt` times meters; default 0.
    pub wprt: R,
    /// The area dependence of `prt`, `pprt`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `prt` times square
    /// meters; default 0.
    pub pprt: R,
    /// The bias-dependent overlap capacitance between the gate and the lightly doped
    /// source, `cgsl`, in farads per meter; default 0.
    pub cgsl: R,
    /// The length dependence of `cgsl`, `lcgsl`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `cgsl` times meters; default 0.
    pub lcgsl: R,
    /// The width dependence of `cgsl`, `wcgsl`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `cgsl` times meters; default 0.
    pub wcgsl: R,
    /// The area dependence of `cgsl`, `pcgsl`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `cgsl` times square
    /// meters; default 0.
    pub pcgsl: R,
    /// The bias-dependent overlap capacitance between the gate and the lightly doped
    /// drain, `cgdl`, in farads per meter; default 0.
    pub cgdl: R,
    /// The length dependence of `cgdl`, `lcgdl`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `cgdl` times meters; default 0.
    pub lcgdl: R,
    /// The width dependence of `cgdl`, `wcgdl`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `cgdl` times meters; default 0.
    pub wcgdl: R,
    /// The area dependence of `cgdl`, `pcgdl`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `cgdl` times square
    /// meters; default 0.
    pub pcgdl: R,
    /// The coefficient of the bias-dependent overlap capacitances, `ckappa`, in volts;
    /// default 0.6.
    pub ckappa: R,
    /// The length dependence of `ckappa`, `lckappa`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `ckappa` times meters; default 0.
    pub lckappa: R,
    /// The width dependence of `ckappa`, `wckappa`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `ckappa` times meters; default 0.
    pub wckappa: R,
    /// The area dependence of `ckappa`, `pckappa`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `ckappa` times
    /// square meters; default 0.
    pub pckappa: R,
    /// The fringing capacitance per width, `cf`, in farads per meter; default 2 · εox /
    /// π · ln(1 + 0.4 µm / `tox`).
    pub cf: Option<R>,
    /// The length dependence of `cf`, `lcf`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `cf` times meters; default 0.
    pub lcf: R,
    /// The width dependence of `cf`, `wcf`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `cf` times meters; default 0.
    pub wcf: R,
    /// The area dependence of `cf`, `pcf`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `cf` times square
    /// meters; default 0.
    pub pcf: R,
    /// The first coefficient of the short-channel effect on the charges, `clc`, in
    /// meters; default 1e-7.
    pub clc: R,
    /// The length dependence of `clc`, `lclc`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `clc` times meters; default 0.
    pub lclc: R,
    /// The width dependence of `clc`, `wclc`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `clc` times meters; default 0.
    pub wclc: R,
    /// The area dependence of `clc`, `pclc`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `clc` times square
    /// meters; default 0.
    pub pclc: R,
    /// The second coefficient of the short-channel effect on the charges, `cle`;
    /// default 0.6.
    pub cle: R,
    /// The length dependence of `cle`, `lcle`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `cle` times meters; default 0.
    pub lcle: R,
    /// The width dependence of `cle`, `wcle`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `cle` times meters; default 0.
    pub wcle: R,
    /// The area dependence of `cle`, `pcle`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `cle` times square
    /// meters; default 0.
    pub pcle: R,
    /// The flat-band voltage of `capmod` 0, `vfbcv`, in volts; default -1.
    pub vfbcv: R,
    /// The length dependence of `vfbcv`, `lvfbcv`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `vfbcv` times meters; default 0.
    pub lvfbcv: R,
    /// The width dependence of `vfbcv`, `wvfbcv`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `vfbcv` times meters; default 0.
    pub wvfbcv: R,
    /// The area dependence of `vfbcv`, `pvfbcv`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `vfbcv` times
    /// square meters; default 0.
    pub pvfbcv: R,
    /// The exponential coefficient of the charge layer's thickness, `capmod` 3's,
    /// `acde`, in meters per volt; default 1.
    pub acde: R,
    /// The length dependence of `acde`, `lacde`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `acde` times meters; default 0.
    pub lacde: R,
    /// The width dependence of `acde`, `wacde`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `acde` times meters; default 0.
    pub wacde: R,
    /// The area dependence of `acde`, `pacde`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `acde` times square
    /// meters; default 0.
    pub pacde: R,
    /// The gate bias's effect on the surface potential, `capmod` 3's, `moin`; default
    /// 15.
    pub moin: R,
    /// The length dependence of `moin`, `lmoin`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `moin` times meters; default 0.
    pub lmoin: R,
    /// The width dependence of `moin`, `wmoin`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `moin` times meters; default 0.
    pub wmoin: R,
    /// The area dependence of `moin`, `pmoin`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `moin` times square
    /// meters; default 0.
    pub pmoin: R,
    /// The subthreshold swing's coefficient in the charges, `capmod` 1 to 3's, `noff`;
    /// default 1.
    pub noff: R,
    /// The length dependence of `noff`, `lnoff`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `noff` times meters; default 0.
    pub lnoff: R,
    /// The width dependence of `noff`, `wnoff`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `noff` times meters; default 0.
    pub wnoff: R,
    /// The area dependence of `noff`, `pnoff`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `noff` times square
    /// meters; default 0.
    pub pnoff: R,
    /// The offset of the threshold in the charges, `capmod` 1 to 3's, `voffcv`, in
    /// volts; default 0.
    pub voffcv: R,
    /// The length dependence of `voffcv`, `lvoffcv`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `voffcv` times meters; default 0.
    pub lvoffcv: R,
    /// The width dependence of `voffcv`, `wvoffcv`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `voffcv` times meters; default 0.
    pub wvoffcv: R,
    /// The area dependence of `voffcv`, `pvoffcv`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `voffcv` times
    /// square meters; default 0.
    pub pvoffcv: R,
    /// The gate-source overlap capacitance per width, beyond the bias-dependent one,
    /// `cgso`, in farads per meter; default `dlc` · Cox − `cgsl` with a positive `dlc`,
    /// else 0.6 · `xj` · Cox.
    pub cgso: Option<R>,
    /// The gate-drain overlap capacitance per width, beyond the bias-dependent one,
    /// `cgdo`, in farads per meter; default `dlc` · Cox − `cgdl` with a positive `dlc`,
    /// else 0.6 · `xj` · Cox.
    pub cgdo: Option<R>,
    /// The gate-bulk overlap capacitance per length, `cgbo`, in farads per meter;
    /// default 2 · `dwc` · Cox.
    pub cgbo: Option<R>,
    /// The length each side of the channel loses, which shortens it, `lint`, in meters;
    /// default 0.
    pub lint: R,
    /// The length dependence of the length lost, `ll`, in meters to the power 1 +
    /// `lln`; default 0.
    pub ll: R,
    /// The length exponent of `ll`, `lln`; default 1.
    pub lln: R,
    /// The width dependence of the length lost, `lw`, in meters times meters to the
    /// power `lwn`; default 0.
    pub lw: R,
    /// The width exponent of `lw`, `lwn`; default 1.
    pub lwn: R,
    /// The area dependence of the length lost, `lwl`, in meters times meters to the
    /// power `lln` + `lwn`; default 0.
    pub lwl: R,
    /// The length dependence of the length the charges lose, `llc`, in meters to the
    /// power 1 + `lln`; default `ll`.
    pub llc: Option<R>,
    /// The width dependence of the length the charges lose, `lwc`, in meters times
    /// meters to the power `lwn`; default `lw`.
    pub lwc: Option<R>,
    /// The area dependence of the length the charges lose, `lwlc`, in meters times
    /// meters to the power `lln` + `lwn`; default `lwl`.
    pub lwlc: Option<R>,
    /// The width each side of the channel loses, which narrows it, `wint`, in meters;
    /// default 0.
    pub wint: R,
    /// The length dependence of the width lost, `wl`, in meters times meters to the
    /// power `wln`; default 0.
    pub wl: R,
    /// The length exponent of `wl`, `wln`; default 1.
    pub wln: R,
    /// The width dependence of the width lost, `ww`, in meters to the power 1 + `wwn`;
    /// default 0.
    pub ww: R,
    /// The width exponent of `ww`, `wwn`; default 1.
    pub wwn: R,
    /// The area dependence of the width lost, `wwl`, in meters times meters to the
    /// power `wln` + `wwn`; default 0.
    pub wwl: R,
    /// The length dependence of the width the charges lose, `wlc`, in meters times
    /// meters to the power `wln`; default `wl`.
    pub wlc: Option<R>,
    /// The width dependence of the width the charges lose, `wwc`, in meters to the
    /// power 1 + `wwn`; default `ww`.
    pub wwc: Option<R>,
    /// The area dependence of the width the charges lose, `wwlc`, in meters times
    /// meters to the power `wln` + `wwn`; default `wwl`.
    pub wwlc: Option<R>,
    /// The length each side of the channel loses in the charges, `dlc`, in meters;
    /// default `lint`.
    pub dlc: Option<R>,
    /// The width each side of the channel loses in the charges, `dwc`, in meters;
    /// default `wint`.
    pub dwc: Option<R>,
    /// The length the mask and etching add, `xl`, in meters; default 0.
    pub xl: R,
    /// The width the mask and etching add, `xw`, in meters; default 0.
    pub xw: R,
    /// The source and drain diffusions' sheet resistance, `rsh`, in ohms per square;
    /// default 0.
    pub rsh: R,
    /// The junctions' saturation current per area, `js`, in amperes per square meter;
    /// default 0.0001.
    pub js: R,
    /// The junctions' saturation current per length of perimeter, `jsw`, in amperes per
    /// meter; default 0.
    pub jsw: R,
    /// The junction current above which the diodes' exponential continues linearly,
    /// zero for none, `ijth`, in amperes; default 0.1.
    pub ijth: R,
    /// The junctions' emission coefficient, `nj`; default 1.
    pub nj: R,
    /// The junctions' saturation current temperature exponent, `xti`; default 3.
    pub xti: R,
    /// The junctions' zero-bias bottom capacitance per area, `cj`, in farads per square
    /// meter; default 0.0005.
    pub cj: R,
    /// The junctions' zero-bias sidewall capacitance per length along the field oxide,
    /// `cjsw`, in farads per meter; default 5e-10.
    pub cjsw: R,
    /// The junctions' zero-bias sidewall capacitance per length along the gate,
    /// `cjswg`, in farads per meter; default `cjsw`.
    pub cjswg: Option<R>,
    /// The bottom capacitance's grading coefficient, `mj`; default 0.5.
    pub mj: R,
    /// The field-oxide sidewall's grading coefficient, `mjsw`; default 0.33.
    pub mjsw: R,
    /// The gate sidewall's grading coefficient, `mjswg`; default `mjsw`.
    pub mjswg: Option<R>,
    /// The bottom junction's built-in potential, at least 0.1 V, `pb`, in volts;
    /// default 1.
    pub pb: R,
    /// The field-oxide sidewall's built-in potential, at least 0.1 V, `pbsw`, in volts;
    /// default 1.
    pub pbsw: R,
    /// The gate sidewall's built-in potential, at least 0.1 V, `pbswg`, in volts;
    /// default `pbsw`.
    pub pbswg: Option<R>,
    /// The temperature coefficient of `pb`, `tpb`, in volts per kelvin; default 0.
    pub tpb: R,
    /// The temperature coefficient of `pbsw`, `tpbsw`, in volts per kelvin; default 0.
    pub tpbsw: R,
    /// The temperature coefficient of `pbswg`, `tpbswg`, in volts per kelvin; default
    /// 0.
    pub tpbswg: R,
    /// The temperature coefficient of `cj`, `tcj`, per kelvin; default 0.
    pub tcj: R,
    /// The temperature coefficient of `cjsw`, `tcjsw`, per kelvin; default 0.
    pub tcjsw: R,
    /// The temperature coefficient of `cjswg`, `tcjswg`, per kelvin; default 0.
    pub tcjswg: R,
    /// The first oxide trap density coefficient of the flicker noise, `noimod` 2, 3 and
    /// 6's, `noia`, in the manual's unit; default 1e20 for an n-channel, 9.9e18 for a
    /// p-channel.
    pub noia: Option<R>,
    /// The second oxide trap density coefficient of the flicker noise, `noib`, in the
    /// manual's unit; default 5e4 for an n-channel, 2.4e3 for a p-channel.
    pub noib: Option<R>,
    /// The third oxide trap density coefficient of the flicker noise, `noic`, in the
    /// manual's unit; default -1.4e-12 for an n-channel, 1.4e-12 for a p-channel.
    pub noic: Option<R>,
    /// The saturation field in the flicker noise, `em`, in volts per meter; default
    /// 41000000.
    pub em: R,
    /// The flicker noise's frequency exponent, `ef`; default 1.
    pub ef: R,
    /// The flicker noise's current exponent, `noimod` 1, 4 and 5's, `af`; default 1.
    pub af: R,
    /// The flicker noise's coefficient, `noimod` 1, 4 and 5's, `kf`; default 0.
    pub kf: R,
    /// The length each side of the channel loses in the flicker noise, beyond `lint`,
    /// `lintnoi`, in meters; default 0.
    pub lintnoi: R,
}

impl<R: Real> Default for Bsim3Parameters<R> {
    #[expect(
        clippy::too_many_lines,
        reason = "one default for each of the card's parameters"
    )]
    fn default() -> Self {
        Self {
            channel: Channel::N,
            width: R::constant(5e-6),
            length: R::constant(5e-6),
            multiplier: R::constant(1.0),
            drain_area: R::constant(0.0),
            source_area: R::constant(0.0),
            drain_perimeter: R::constant(0.0),
            source_perimeter: R::constant(0.0),
            drain_squares: R::constant(1.0),
            source_squares: R::constant(1.0),
            threshold_adjustment: R::constant(0.0),
            mobility_multiplier: R::constant(1.0),
            mobility_model: Mobility::One,
            capacitance_model: Charges::Three,
            noise_model: NoiseModel::One,
            check_parameters: false,
            charge_partition: 0.0,
            nominal_temperature: R::constant(REFERENCE_TEMPERATURE),
            tox: R::constant(1.5e-8),
            toxm: None,
            xj: R::constant(1.5e-7),
            lxj: R::constant(0.0),
            wxj: R::constant(0.0),
            pxj: R::constant(0.0),
            nch: None,
            lnch: R::constant(0.0),
            wnch: R::constant(0.0),
            pnch: R::constant(0.0),
            nsub: R::constant(6.0e22),
            lnsub: R::constant(0.0),
            wnsub: R::constant(0.0),
            pnsub: R::constant(0.0),
            ngate: R::constant(0.0),
            lngate: R::constant(0.0),
            wngate: R::constant(0.0),
            pngate: R::constant(0.0),
            gamma1: None,
            lgamma1: R::constant(0.0),
            wgamma1: R::constant(0.0),
            pgamma1: R::constant(0.0),
            gamma2: None,
            lgamma2: R::constant(0.0),
            wgamma2: R::constant(0.0),
            pgamma2: R::constant(0.0),
            vbx: None,
            lvbx: R::constant(0.0),
            wvbx: R::constant(0.0),
            pvbx: R::constant(0.0),
            vbm: R::constant(-3.0),
            lvbm: R::constant(0.0),
            wvbm: R::constant(0.0),
            pvbm: R::constant(0.0),
            xt: R::constant(1.55e-7),
            lxt: R::constant(0.0),
            wxt: R::constant(0.0),
            pxt: R::constant(0.0),
            vth0: None,
            lvth0: R::constant(0.0),
            wvth0: R::constant(0.0),
            pvth0: R::constant(0.0),
            vfb: None,
            lvfb: R::constant(0.0),
            wvfb: R::constant(0.0),
            pvfb: R::constant(0.0),
            k1: None,
            lk1: R::constant(0.0),
            wk1: R::constant(0.0),
            pk1: R::constant(0.0),
            k2: None,
            lk2: R::constant(0.0),
            wk2: R::constant(0.0),
            pk2: R::constant(0.0),
            k3: R::constant(80.0),
            lk3: R::constant(0.0),
            wk3: R::constant(0.0),
            pk3: R::constant(0.0),
            k3b: R::constant(0.0),
            lk3b: R::constant(0.0),
            wk3b: R::constant(0.0),
            pk3b: R::constant(0.0),
            w0: R::constant(2.5e-6),
            lw0: R::constant(0.0),
            ww0: R::constant(0.0),
            pw0: R::constant(0.0),
            nlx: R::constant(1.74e-7),
            lnlx: R::constant(0.0),
            wnlx: R::constant(0.0),
            pnlx: R::constant(0.0),
            dvt0: R::constant(2.2),
            ldvt0: R::constant(0.0),
            wdvt0: R::constant(0.0),
            pdvt0: R::constant(0.0),
            dvt1: R::constant(0.53),
            ldvt1: R::constant(0.0),
            wdvt1: R::constant(0.0),
            pdvt1: R::constant(0.0),
            dvt2: R::constant(-0.032),
            ldvt2: R::constant(0.0),
            wdvt2: R::constant(0.0),
            pdvt2: R::constant(0.0),
            dvt0w: R::constant(0.0),
            ldvt0w: R::constant(0.0),
            wdvt0w: R::constant(0.0),
            pdvt0w: R::constant(0.0),
            dvt1w: R::constant(5.3e6),
            ldvt1w: R::constant(0.0),
            wdvt1w: R::constant(0.0),
            pdvt1w: R::constant(0.0),
            dvt2w: R::constant(-0.032),
            ldvt2w: R::constant(0.0),
            wdvt2w: R::constant(0.0),
            pdvt2w: R::constant(0.0),
            dsub: None,
            ldsub: R::constant(0.0),
            wdsub: R::constant(0.0),
            pdsub: R::constant(0.0),
            eta0: R::constant(0.08),
            leta0: R::constant(0.0),
            weta0: R::constant(0.0),
            peta0: R::constant(0.0),
            etab: R::constant(-0.07),
            letab: R::constant(0.0),
            wetab: R::constant(0.0),
            petab: R::constant(0.0),
            u0: None,
            lu0: R::constant(0.0),
            wu0: R::constant(0.0),
            pu0: R::constant(0.0),
            ua: R::constant(2.25e-9),
            lua: R::constant(0.0),
            wua: R::constant(0.0),
            pua: R::constant(0.0),
            ub: R::constant(5.87e-19),
            lub: R::constant(0.0),
            wub: R::constant(0.0),
            r#pub: R::constant(0.0),
            uc: None,
            luc: R::constant(0.0),
            wuc: R::constant(0.0),
            puc: R::constant(0.0),
            vsat: R::constant(80000.0),
            lvsat: R::constant(0.0),
            wvsat: R::constant(0.0),
            pvsat: R::constant(0.0),
            a0: R::constant(1.0),
            la0: R::constant(0.0),
            wa0: R::constant(0.0),
            pa0: R::constant(0.0),
            ags: R::constant(0.0),
            lags: R::constant(0.0),
            wags: R::constant(0.0),
            pags: R::constant(0.0),
            a1: R::constant(0.0),
            la1: R::constant(0.0),
            wa1: R::constant(0.0),
            pa1: R::constant(0.0),
            a2: R::constant(1.0),
            la2: R::constant(0.0),
            wa2: R::constant(0.0),
            pa2: R::constant(0.0),
            b0: R::constant(0.0),
            lb0: R::constant(0.0),
            wb0: R::constant(0.0),
            pb0: R::constant(0.0),
            b1: R::constant(0.0),
            lb1: R::constant(0.0),
            wb1: R::constant(0.0),
            pb1: R::constant(0.0),
            keta: R::constant(-0.047),
            lketa: R::constant(0.0),
            wketa: R::constant(0.0),
            pketa: R::constant(0.0),
            voff: R::constant(-0.08),
            lvoff: R::constant(0.0),
            wvoff: R::constant(0.0),
            pvoff: R::constant(0.0),
            nfactor: R::constant(1.0),
            lnfactor: R::constant(0.0),
            wnfactor: R::constant(0.0),
            pnfactor: R::constant(0.0),
            cdsc: R::constant(0.00024),
            lcdsc: R::constant(0.0),
            wcdsc: R::constant(0.0),
            pcdsc: R::constant(0.0),
            cdscb: R::constant(0.0),
            lcdscb: R::constant(0.0),
            wcdscb: R::constant(0.0),
            pcdscb: R::constant(0.0),
            cdscd: R::constant(0.0),
            lcdscd: R::constant(0.0),
            wcdscd: R::constant(0.0),
            pcdscd: R::constant(0.0),
            cit: R::constant(0.0),
            lcit: R::constant(0.0),
            wcit: R::constant(0.0),
            pcit: R::constant(0.0),
            pclm: R::constant(1.3),
            lpclm: R::constant(0.0),
            wpclm: R::constant(0.0),
            ppclm: R::constant(0.0),
            pdiblc1: R::constant(0.39),
            lpdiblc1: R::constant(0.0),
            wpdiblc1: R::constant(0.0),
            ppdiblc1: R::constant(0.0),
            pdiblc2: R::constant(0.0086),
            lpdiblc2: R::constant(0.0),
            wpdiblc2: R::constant(0.0),
            ppdiblc2: R::constant(0.0),
            pdiblcb: R::constant(0.0),
            lpdiblcb: R::constant(0.0),
            wpdiblcb: R::constant(0.0),
            ppdiblcb: R::constant(0.0),
            drout: R::constant(0.56),
            ldrout: R::constant(0.0),
            wdrout: R::constant(0.0),
            pdrout: R::constant(0.0),
            pscbe1: R::constant(4.24e8),
            lpscbe1: R::constant(0.0),
            wpscbe1: R::constant(0.0),
            ppscbe1: R::constant(0.0),
            pscbe2: R::constant(1.0e-5),
            lpscbe2: R::constant(0.0),
            wpscbe2: R::constant(0.0),
            ppscbe2: R::constant(0.0),
            pvag: R::constant(0.0),
            lpvag: R::constant(0.0),
            wpvag: R::constant(0.0),
            ppvag: R::constant(0.0),
            delta: R::constant(0.01),
            ldelta: R::constant(0.0),
            wdelta: R::constant(0.0),
            pdelta: R::constant(0.0),
            rdsw: R::constant(0.0),
            lrdsw: R::constant(0.0),
            wrdsw: R::constant(0.0),
            prdsw: R::constant(0.0),
            prwg: R::constant(0.0),
            lprwg: R::constant(0.0),
            wprwg: R::constant(0.0),
            pprwg: R::constant(0.0),
            prwb: R::constant(0.0),
            lprwb: R::constant(0.0),
            wprwb: R::constant(0.0),
            pprwb: R::constant(0.0),
            wr: R::constant(1.0),
            lwr: R::constant(0.0),
            wwr: R::constant(0.0),
            pwr: R::constant(0.0),
            dwg: R::constant(0.0),
            ldwg: R::constant(0.0),
            wdwg: R::constant(0.0),
            pdwg: R::constant(0.0),
            dwb: R::constant(0.0),
            ldwb: R::constant(0.0),
            wdwb: R::constant(0.0),
            pdwb: R::constant(0.0),
            alpha0: R::constant(0.0),
            lalpha0: R::constant(0.0),
            walpha0: R::constant(0.0),
            palpha0: R::constant(0.0),
            alpha1: R::constant(0.0),
            lalpha1: R::constant(0.0),
            walpha1: R::constant(0.0),
            palpha1: R::constant(0.0),
            beta0: R::constant(30.0),
            lbeta0: R::constant(0.0),
            wbeta0: R::constant(0.0),
            pbeta0: R::constant(0.0),
            kt1: R::constant(-0.11),
            lkt1: R::constant(0.0),
            wkt1: R::constant(0.0),
            pkt1: R::constant(0.0),
            kt1l: R::constant(0.0),
            lkt1l: R::constant(0.0),
            wkt1l: R::constant(0.0),
            pkt1l: R::constant(0.0),
            kt2: R::constant(0.022),
            lkt2: R::constant(0.0),
            wkt2: R::constant(0.0),
            pkt2: R::constant(0.0),
            ute: R::constant(-1.5),
            lute: R::constant(0.0),
            wute: R::constant(0.0),
            pute: R::constant(0.0),
            ua1: R::constant(4.31e-9),
            lua1: R::constant(0.0),
            wua1: R::constant(0.0),
            pua1: R::constant(0.0),
            ub1: R::constant(-7.61e-18),
            lub1: R::constant(0.0),
            wub1: R::constant(0.0),
            pub1: R::constant(0.0),
            uc1: None,
            luc1: R::constant(0.0),
            wuc1: R::constant(0.0),
            puc1: R::constant(0.0),
            at: R::constant(33000.0),
            lat: R::constant(0.0),
            wat: R::constant(0.0),
            pat: R::constant(0.0),
            prt: R::constant(0.0),
            lprt: R::constant(0.0),
            wprt: R::constant(0.0),
            pprt: R::constant(0.0),
            cgsl: R::constant(0.0),
            lcgsl: R::constant(0.0),
            wcgsl: R::constant(0.0),
            pcgsl: R::constant(0.0),
            cgdl: R::constant(0.0),
            lcgdl: R::constant(0.0),
            wcgdl: R::constant(0.0),
            pcgdl: R::constant(0.0),
            ckappa: R::constant(0.6),
            lckappa: R::constant(0.0),
            wckappa: R::constant(0.0),
            pckappa: R::constant(0.0),
            cf: None,
            lcf: R::constant(0.0),
            wcf: R::constant(0.0),
            pcf: R::constant(0.0),
            clc: R::constant(1.0e-7),
            lclc: R::constant(0.0),
            wclc: R::constant(0.0),
            pclc: R::constant(0.0),
            cle: R::constant(0.6),
            lcle: R::constant(0.0),
            wcle: R::constant(0.0),
            pcle: R::constant(0.0),
            vfbcv: R::constant(-1.0),
            lvfbcv: R::constant(0.0),
            wvfbcv: R::constant(0.0),
            pvfbcv: R::constant(0.0),
            acde: R::constant(1.0),
            lacde: R::constant(0.0),
            wacde: R::constant(0.0),
            pacde: R::constant(0.0),
            moin: R::constant(15.0),
            lmoin: R::constant(0.0),
            wmoin: R::constant(0.0),
            pmoin: R::constant(0.0),
            noff: R::constant(1.0),
            lnoff: R::constant(0.0),
            wnoff: R::constant(0.0),
            pnoff: R::constant(0.0),
            voffcv: R::constant(0.0),
            lvoffcv: R::constant(0.0),
            wvoffcv: R::constant(0.0),
            pvoffcv: R::constant(0.0),
            cgso: None,
            cgdo: None,
            cgbo: None,
            lint: R::constant(0.0),
            ll: R::constant(0.0),
            lln: R::constant(1.0),
            lw: R::constant(0.0),
            lwn: R::constant(1.0),
            lwl: R::constant(0.0),
            llc: None,
            lwc: None,
            lwlc: None,
            wint: R::constant(0.0),
            wl: R::constant(0.0),
            wln: R::constant(1.0),
            ww: R::constant(0.0),
            wwn: R::constant(1.0),
            wwl: R::constant(0.0),
            wlc: None,
            wwc: None,
            wwlc: None,
            dlc: None,
            dwc: None,
            xl: R::constant(0.0),
            xw: R::constant(0.0),
            rsh: R::constant(0.0),
            js: R::constant(0.0001),
            jsw: R::constant(0.0),
            ijth: R::constant(0.1),
            nj: R::constant(1.0),
            xti: R::constant(3.0),
            cj: R::constant(0.0005),
            cjsw: R::constant(5.0e-10),
            cjswg: None,
            mj: R::constant(0.5),
            mjsw: R::constant(0.33),
            mjswg: None,
            pb: R::constant(1.0),
            pbsw: R::constant(1.0),
            pbswg: None,
            tpb: R::constant(0.0),
            tpbsw: R::constant(0.0),
            tpbswg: R::constant(0.0),
            tcj: R::constant(0.0),
            tcjsw: R::constant(0.0),
            tcjswg: R::constant(0.0),
            noia: None,
            noib: None,
            noic: None,
            em: R::constant(4.1e7),
            ef: R::constant(1.0),
            af: R::constant(1.0),
            kf: R::constant(0.0),
            lintnoi: R::constant(0.0),
        }
    }
}

/// The binned parameters of one size: each one's value, its length, width and area
/// dependences added. A parameter whose default derives from others is `None` where the
/// card leaves it out, its dependences then read past, as ngspice derives it in their
/// place.
#[derive(Debug, Clone, Copy)]
pub(super) struct Binned<R: Real> {
    pub(super) xj: R,
    pub(super) nch: R,
    pub(super) nsub: R,
    pub(super) ngate: R,
    pub(super) gamma1: Option<R>,
    pub(super) gamma2: Option<R>,
    pub(super) vbx: Option<R>,
    pub(super) vbm: R,
    pub(super) xt: R,
    pub(super) vth0: Option<R>,
    pub(super) vfb: Option<R>,
    pub(super) k1: Option<R>,
    pub(super) k2: Option<R>,
    pub(super) k3: R,
    pub(super) k3b: R,
    pub(super) w0: R,
    pub(super) nlx: R,
    pub(super) dvt0: R,
    pub(super) dvt1: R,
    pub(super) dvt2: R,
    pub(super) dvt0w: R,
    pub(super) dvt1w: R,
    pub(super) dvt2w: R,
    pub(super) dsub: R,
    pub(super) eta0: R,
    pub(super) etab: R,
    pub(super) u0: R,
    pub(super) ua: R,
    pub(super) ub: R,
    pub(super) uc: R,
    pub(super) vsat: R,
    pub(super) a0: R,
    pub(super) ags: R,
    pub(super) a1: R,
    pub(super) a2: R,
    pub(super) b0: R,
    pub(super) b1: R,
    pub(super) keta: R,
    pub(super) voff: R,
    pub(super) nfactor: R,
    pub(super) cdsc: R,
    pub(super) cdscb: R,
    pub(super) cdscd: R,
    pub(super) cit: R,
    pub(super) pclm: R,
    pub(super) pdiblc1: R,
    pub(super) pdiblc2: R,
    pub(super) pdiblcb: R,
    pub(super) drout: R,
    pub(super) pscbe1: R,
    pub(super) pscbe2: R,
    pub(super) pvag: R,
    pub(super) delta: R,
    pub(super) rdsw: R,
    pub(super) prwg: R,
    pub(super) prwb: R,
    pub(super) wr: R,
    pub(super) dwg: R,
    pub(super) dwb: R,
    pub(super) alpha0: R,
    pub(super) alpha1: R,
    pub(super) beta0: R,
    pub(super) kt1: R,
    pub(super) kt1l: R,
    pub(super) kt2: R,
    pub(super) ute: R,
    pub(super) ua1: R,
    pub(super) ub1: R,
    pub(super) uc1: R,
    pub(super) at: R,
    pub(super) prt: R,
    pub(super) cgsl: R,
    pub(super) cgdl: R,
    pub(super) ckappa: R,
    pub(super) cf: R,
    pub(super) clc: R,
    pub(super) cle: R,
    pub(super) vfbcv: R,
    pub(super) acde: R,
    pub(super) moin: R,
    pub(super) noff: R,
    pub(super) voffcv: R,
}

impl<R: Real> Bsim3Parameters<R> {
    /// The binned parameters at the reciprocals of the effective length and width, and
    /// of their product.
    #[expect(
        clippy::too_many_lines,
        reason = "one sum for each of the binned parameters"
    )]
    pub(super) fn binned(
        &self,
        [inverse_length, inverse_width, inverse_area]: [R; 3],
    ) -> Binned<R> {
        Binned {
            xj: self.xj
                + self.lxj * inverse_length
                + self.wxj * inverse_width
                + self.pxj * inverse_area,
            nch: self.nch.unwrap_or(R::constant(1.7e23))
                + self.lnch * inverse_length
                + self.wnch * inverse_width
                + self.pnch * inverse_area,
            nsub: self.nsub
                + self.lnsub * inverse_length
                + self.wnsub * inverse_width
                + self.pnsub * inverse_area,
            ngate: self.ngate
                + self.lngate * inverse_length
                + self.wngate * inverse_width
                + self.pngate * inverse_area,
            gamma1: self.gamma1.map(|value| {
                value
                    + self.lgamma1 * inverse_length
                    + self.wgamma1 * inverse_width
                    + self.pgamma1 * inverse_area
            }),
            gamma2: self.gamma2.map(|value| {
                value
                    + self.lgamma2 * inverse_length
                    + self.wgamma2 * inverse_width
                    + self.pgamma2 * inverse_area
            }),
            vbx: self.vbx.map(|value| {
                value
                    + self.lvbx * inverse_length
                    + self.wvbx * inverse_width
                    + self.pvbx * inverse_area
            }),
            vbm: self.vbm
                + self.lvbm * inverse_length
                + self.wvbm * inverse_width
                + self.pvbm * inverse_area,
            xt: self.xt
                + self.lxt * inverse_length
                + self.wxt * inverse_width
                + self.pxt * inverse_area,
            vth0: self.vth0.map(|value| {
                value
                    + self.lvth0 * inverse_length
                    + self.wvth0 * inverse_width
                    + self.pvth0 * inverse_area
            }),
            vfb: self.vfb.map(|value| {
                value
                    + self.lvfb * inverse_length
                    + self.wvfb * inverse_width
                    + self.pvfb * inverse_area
            }),
            k1: self.k1.map(|value| {
                value
                    + self.lk1 * inverse_length
                    + self.wk1 * inverse_width
                    + self.pk1 * inverse_area
            }),
            k2: self.k2.map(|value| {
                value
                    + self.lk2 * inverse_length
                    + self.wk2 * inverse_width
                    + self.pk2 * inverse_area
            }),
            k3: self.k3
                + self.lk3 * inverse_length
                + self.wk3 * inverse_width
                + self.pk3 * inverse_area,
            k3b: self.k3b
                + self.lk3b * inverse_length
                + self.wk3b * inverse_width
                + self.pk3b * inverse_area,
            w0: self.w0
                + self.lw0 * inverse_length
                + self.ww0 * inverse_width
                + self.pw0 * inverse_area,
            nlx: self.nlx
                + self.lnlx * inverse_length
                + self.wnlx * inverse_width
                + self.pnlx * inverse_area,
            dvt0: self.dvt0
                + self.ldvt0 * inverse_length
                + self.wdvt0 * inverse_width
                + self.pdvt0 * inverse_area,
            dvt1: self.dvt1
                + self.ldvt1 * inverse_length
                + self.wdvt1 * inverse_width
                + self.pdvt1 * inverse_area,
            dvt2: self.dvt2
                + self.ldvt2 * inverse_length
                + self.wdvt2 * inverse_width
                + self.pdvt2 * inverse_area,
            dvt0w: self.dvt0w
                + self.ldvt0w * inverse_length
                + self.wdvt0w * inverse_width
                + self.pdvt0w * inverse_area,
            dvt1w: self.dvt1w
                + self.ldvt1w * inverse_length
                + self.wdvt1w * inverse_width
                + self.pdvt1w * inverse_area,
            dvt2w: self.dvt2w
                + self.ldvt2w * inverse_length
                + self.wdvt2w * inverse_width
                + self.pdvt2w * inverse_area,
            dsub: self.dsub.unwrap_or(self.drout)
                + self.ldsub * inverse_length
                + self.wdsub * inverse_width
                + self.pdsub * inverse_area,
            eta0: self.eta0
                + self.leta0 * inverse_length
                + self.weta0 * inverse_width
                + self.peta0 * inverse_area,
            etab: self.etab
                + self.letab * inverse_length
                + self.wetab * inverse_width
                + self.petab * inverse_area,
            u0: self
                .u0
                .unwrap_or(R::constant(by_channel(self.channel, 0.067, 0.025)))
                + self.lu0 * inverse_length
                + self.wu0 * inverse_width
                + self.pu0 * inverse_area,
            ua: self.ua
                + self.lua * inverse_length
                + self.wua * inverse_width
                + self.pua * inverse_area,
            ub: self.ub
                + self.lub * inverse_length
                + self.wub * inverse_width
                + self.r#pub * inverse_area,
            uc: self
                .uc
                .unwrap_or(R::constant(self.mobility_model.select(-0.0465e-9, -0.0465)))
                + self.luc * inverse_length
                + self.wuc * inverse_width
                + self.puc * inverse_area,
            vsat: self.vsat
                + self.lvsat * inverse_length
                + self.wvsat * inverse_width
                + self.pvsat * inverse_area,
            a0: self.a0
                + self.la0 * inverse_length
                + self.wa0 * inverse_width
                + self.pa0 * inverse_area,
            ags: self.ags
                + self.lags * inverse_length
                + self.wags * inverse_width
                + self.pags * inverse_area,
            a1: self.a1
                + self.la1 * inverse_length
                + self.wa1 * inverse_width
                + self.pa1 * inverse_area,
            a2: self.a2
                + self.la2 * inverse_length
                + self.wa2 * inverse_width
                + self.pa2 * inverse_area,
            b0: self.b0
                + self.lb0 * inverse_length
                + self.wb0 * inverse_width
                + self.pb0 * inverse_area,
            b1: self.b1
                + self.lb1 * inverse_length
                + self.wb1 * inverse_width
                + self.pb1 * inverse_area,
            keta: self.keta
                + self.lketa * inverse_length
                + self.wketa * inverse_width
                + self.pketa * inverse_area,
            voff: self.voff
                + self.lvoff * inverse_length
                + self.wvoff * inverse_width
                + self.pvoff * inverse_area,
            nfactor: self.nfactor
                + self.lnfactor * inverse_length
                + self.wnfactor * inverse_width
                + self.pnfactor * inverse_area,
            cdsc: self.cdsc
                + self.lcdsc * inverse_length
                + self.wcdsc * inverse_width
                + self.pcdsc * inverse_area,
            cdscb: self.cdscb
                + self.lcdscb * inverse_length
                + self.wcdscb * inverse_width
                + self.pcdscb * inverse_area,
            cdscd: self.cdscd
                + self.lcdscd * inverse_length
                + self.wcdscd * inverse_width
                + self.pcdscd * inverse_area,
            cit: self.cit
                + self.lcit * inverse_length
                + self.wcit * inverse_width
                + self.pcit * inverse_area,
            pclm: self.pclm
                + self.lpclm * inverse_length
                + self.wpclm * inverse_width
                + self.ppclm * inverse_area,
            pdiblc1: self.pdiblc1
                + self.lpdiblc1 * inverse_length
                + self.wpdiblc1 * inverse_width
                + self.ppdiblc1 * inverse_area,
            pdiblc2: self.pdiblc2
                + self.lpdiblc2 * inverse_length
                + self.wpdiblc2 * inverse_width
                + self.ppdiblc2 * inverse_area,
            pdiblcb: self.pdiblcb
                + self.lpdiblcb * inverse_length
                + self.wpdiblcb * inverse_width
                + self.ppdiblcb * inverse_area,
            drout: self.drout
                + self.ldrout * inverse_length
                + self.wdrout * inverse_width
                + self.pdrout * inverse_area,
            pscbe1: self.pscbe1
                + self.lpscbe1 * inverse_length
                + self.wpscbe1 * inverse_width
                + self.ppscbe1 * inverse_area,
            pscbe2: self.pscbe2
                + self.lpscbe2 * inverse_length
                + self.wpscbe2 * inverse_width
                + self.ppscbe2 * inverse_area,
            pvag: self.pvag
                + self.lpvag * inverse_length
                + self.wpvag * inverse_width
                + self.ppvag * inverse_area,
            delta: self.delta
                + self.ldelta * inverse_length
                + self.wdelta * inverse_width
                + self.pdelta * inverse_area,
            rdsw: self.rdsw
                + self.lrdsw * inverse_length
                + self.wrdsw * inverse_width
                + self.prdsw * inverse_area,
            prwg: self.prwg
                + self.lprwg * inverse_length
                + self.wprwg * inverse_width
                + self.pprwg * inverse_area,
            prwb: self.prwb
                + self.lprwb * inverse_length
                + self.wprwb * inverse_width
                + self.pprwb * inverse_area,
            wr: self.wr
                + self.lwr * inverse_length
                + self.wwr * inverse_width
                + self.pwr * inverse_area,
            dwg: self.dwg
                + self.ldwg * inverse_length
                + self.wdwg * inverse_width
                + self.pdwg * inverse_area,
            dwb: self.dwb
                + self.ldwb * inverse_length
                + self.wdwb * inverse_width
                + self.pdwb * inverse_area,
            alpha0: self.alpha0
                + self.lalpha0 * inverse_length
                + self.walpha0 * inverse_width
                + self.palpha0 * inverse_area,
            alpha1: self.alpha1
                + self.lalpha1 * inverse_length
                + self.walpha1 * inverse_width
                + self.palpha1 * inverse_area,
            beta0: self.beta0
                + self.lbeta0 * inverse_length
                + self.wbeta0 * inverse_width
                + self.pbeta0 * inverse_area,
            kt1: self.kt1
                + self.lkt1 * inverse_length
                + self.wkt1 * inverse_width
                + self.pkt1 * inverse_area,
            kt1l: self.kt1l
                + self.lkt1l * inverse_length
                + self.wkt1l * inverse_width
                + self.pkt1l * inverse_area,
            kt2: self.kt2
                + self.lkt2 * inverse_length
                + self.wkt2 * inverse_width
                + self.pkt2 * inverse_area,
            ute: self.ute
                + self.lute * inverse_length
                + self.wute * inverse_width
                + self.pute * inverse_area,
            ua1: self.ua1
                + self.lua1 * inverse_length
                + self.wua1 * inverse_width
                + self.pua1 * inverse_area,
            ub1: self.ub1
                + self.lub1 * inverse_length
                + self.wub1 * inverse_width
                + self.pub1 * inverse_area,
            uc1: self
                .uc1
                .unwrap_or(R::constant(self.mobility_model.select(-0.056e-9, -0.056)))
                + self.luc1 * inverse_length
                + self.wuc1 * inverse_width
                + self.puc1 * inverse_area,
            at: self.at
                + self.lat * inverse_length
                + self.wat * inverse_width
                + self.pat * inverse_area,
            prt: self.prt
                + self.lprt * inverse_length
                + self.wprt * inverse_width
                + self.pprt * inverse_area,
            cgsl: self.cgsl
                + self.lcgsl * inverse_length
                + self.wcgsl * inverse_width
                + self.pcgsl * inverse_area,
            cgdl: self.cgdl
                + self.lcgdl * inverse_length
                + self.wcgdl * inverse_width
                + self.pcgdl * inverse_area,
            ckappa: self.ckappa
                + self.lckappa * inverse_length
                + self.wckappa * inverse_width
                + self.pckappa * inverse_area,
            cf: self.cf.unwrap_or(fringe(self.tox))
                + self.lcf * inverse_length
                + self.wcf * inverse_width
                + self.pcf * inverse_area,
            clc: self.clc
                + self.lclc * inverse_length
                + self.wclc * inverse_width
                + self.pclc * inverse_area,
            cle: self.cle
                + self.lcle * inverse_length
                + self.wcle * inverse_width
                + self.pcle * inverse_area,
            vfbcv: self.vfbcv
                + self.lvfbcv * inverse_length
                + self.wvfbcv * inverse_width
                + self.pvfbcv * inverse_area,
            acde: self.acde
                + self.lacde * inverse_length
                + self.wacde * inverse_width
                + self.pacde * inverse_area,
            moin: self.moin
                + self.lmoin * inverse_length
                + self.wmoin * inverse_width
                + self.pmoin * inverse_area,
            noff: self.noff
                + self.lnoff * inverse_length
                + self.wnoff * inverse_width
                + self.pnoff * inverse_area,
            voffcv: self.voffcv
                + self.lvoffcv * inverse_length
                + self.wvoffcv * inverse_width
                + self.pvoffcv * inverse_area,
        }
    }
}
