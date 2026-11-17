//! The parameters of a BSIM4 transistor, by the names of the BSIM4.5 manual, in SI units,
//! with ngspice's defaults.
//!
//! The defaults are those of ngspice 47's `b4v5set.c`; a parameter whose default derives
//! from others, as `vth0` from `vfb`, is optional, and its documentation says what it
//! derives from.
//!
//! Part of jolty's BSIM4, derived from BSIM4.5.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2005 Regents of the University of California. All rights reserved. Used
//! under the BSIM4 terms of use, which the repository's `NOTICE` reproduces.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{
    BodyResistance, DrainSourceResistance, FlickerNoise, GateResistance, Perimeter, ThermalNoise,
    fringe,
};
use crate::kernel::bsim3::by_channel;
use crate::kernel::jfet::Channel;
use crate::{Lift, REFERENCE_TEMPERATURE, Real};

/// The parameters of a [`Bsim4`](super::Bsim4): the instance's geometry, then the model's
/// card. Each parameter of the card keeps the name the BSIM4.5 manual gives it, and each
/// one that bins has a length, a width and an area dependence, `l`, `w` and `p` before its
/// name, in SI units: a dependence on the reciprocal of the effective length in meters,
/// not micrometers.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
#[serde(default, deny_unknown_fields, bound = "")]
pub struct Bsim4Parameters<R: Real = f64> {
    /// The channel; default `n`.
    pub channel: Channel,
    /// The channel's drawn width, `w`, in meters, of all its fingers together; default
    /// 5 µm.
    pub width: R,
    /// The channel's drawn length, `l`, in meters; default 5 µm.
    pub length: R,
    /// The number of identical transistors in parallel, `m`; default 1.
    pub multiplier: R,
    /// The number of fingers the channel's width is split into, `nf`; default 1.
    pub fingers: R,
    /// The drain diffusion's area, `ad`, in square meters; default derived from the
    /// fingers and `dmcg`, `dmci` and `dmcgt`.
    pub drain_area: Option<R>,
    /// The source diffusion's area, `as`, in square meters; default derived from the
    /// fingers and `dmcg`, `dmci` and `dmcgt`.
    pub source_area: Option<R>,
    /// The drain diffusion's perimeter, `pd`, in meters; default derived from the fingers
    /// and `dmcg`, `dmci` and `dmcgt`.
    pub drain_perimeter: Option<R>,
    /// The source diffusion's perimeter, `ps`, in meters; default derived from the fingers
    /// and `dmcg`, `dmci` and `dmcgt`.
    pub source_perimeter: Option<R>,
    /// The drain diffusion's length in squares, `nrd`, behind which `rsh` sets a
    /// resistance where it is positive; default none, and no resistance.
    pub drain_squares: Option<R>,
    /// The source diffusion's length in squares, `nrs`, behind which `rsh` sets a
    /// resistance where it is positive; default none, and no resistance.
    pub source_squares: Option<R>,
    /// The distance from the gate to the edge of the diffusion on one side, `sa`, in
    /// meters, which sets the stress of the length of diffusion where it and `sb` are
    /// positive; default 0.
    pub diffusion_edge_a: R,
    /// The distance from the gate to the edge of the diffusion on the other side, `sb`,
    /// in meters; default 0.
    pub diffusion_edge_b: R,
    /// The distance between neighbouring fingers, `sd`, in meters; default twice `dmcg`.
    pub finger_spacing: Option<R>,
    /// A shift of the zero-bias threshold, `delvto`, in volts; default 0.
    pub threshold_adjustment: R,
    /// A factor of the low-field mobility, `mulu0`; default 1.
    pub mobility_multiplier: R,
    /// The body's resistance network, `rbodymod`, 0 or 1; default 0.
    pub body_resistance_model: BodyResistance,
    /// The gate's resistance network, `rgatemod`, 0 to 3; default 0.
    pub gate_resistance_model: GateResistance,
    /// What the diffusions' perimeters `ps` and `pd` include, `permod`, 0 or 1; default
    /// 1.
    pub perimeter_model: Perimeter,
    /// The source and drain resistances, `rdsmod`, 0 or 1; default 0.
    pub drain_source_resistance_model: DrainSourceResistance,
    /// The flicker noise model, `fnoimod`, 0 or 1; default 1.
    pub flicker_noise_model: FlickerNoise,
    /// The channel's thermal noise model, `tnoimod`, 0 or 1; default 0.
    pub thermal_noise_model: ThermalNoise,
    /// Whether the size-dependent parameters are checked and clamped where they lie out
    /// of range, `paramchk = 1`; default true.
    pub check_parameters: bool,
    /// The share of the channel's charge the drain takes in saturation, `xpart`: 0.4 below
    /// 0.5, 0.5 at 0.5 and none above it; default 0.
    pub charge_partition: f64,
    /// The temperature the parameters were measured at, `tnom`, in kelvins; default 27 °C.
    pub nominal_temperature: R,
    /// The electrical thickness of the gate oxide, `toxe`, in meters; default 3 nm, or
    /// `toxp` plus `dtox` where `toxp` alone is given.
    pub toxe: Option<R>,
    /// The physical thickness of the gate oxide, `toxp`, in meters; default `toxe` less
    /// `dtox` where `toxe` is given, else `toxe`.
    pub toxp: Option<R>,
    /// The oxide thickness the parameters were extracted at, `toxm`, in meters; default
    /// `toxe` where it is given, else 3 nm.
    pub toxm: Option<R>,
    /// The oxide thickness of reference of the gate tunneling currents, `toxref`, in
    /// meters; default 3e-09.
    pub toxref: R,
    /// The electrical less the physical thickness of the gate oxide, `dtox`, in meters;
    /// default 0.
    pub dtox: R,
    /// The dielectric constant of the gate oxide relative to vacuum, `epsrox`; default
    /// 3.9.
    pub epsrox: R,
    /// The coupling capacitance of the drain and the source to the channel, `cdsc`, in
    /// farads per square meter; default 0.00024.
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
    /// The body-bias dependence of `cdsc`, `cdscb`, in farads per volt square meter;
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
    /// The drain-bias dependence of `cdsc`, `cdscd`, in farads per volt square meter;
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
    /// The interface state capacitance, `cit`, in farads per square meter; default 0.
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
    /// The subthreshold swing coefficient, `nfactor`; default 1.
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
    /// The source and drain junctions' depth, `xj`, in meters; default 1.5e-07.
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
    /// The saturation velocity at tnom, `vsat`, in meters per second; default 80000.
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
    /// The temperature coefficient of vsat, `at`, in meters per second; default 33000.
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
    /// The non-uniform depletion width effect coefficient, `a0`; default 1.
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
    /// The gate bias coefficient of Abulk, `ags`, in per volt; default 0.
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
    /// The non-saturation effect coefficient, `a1`, in per volt; default 0.
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
    /// The non-saturation effect coefficient, `a2`; default 1.
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
    /// The body-bias coefficient of the non-uniform depletion width effect, `keta`, in
    /// per volt; default -0.047.
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
    /// The substrate doping concentration, `nsub`, in per cubic meter; default 6e+22.
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
    /// The channel doping concentration at the depletion edge, `ndep`, in per cubic
    /// meter; default 1.7e23 per cubic meter, or derived from `gamma1` where it alone
    /// is given.
    pub ndep: Option<R>,
    /// The length dependence of `ndep`, `lndep`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `ndep` times meters; default 0.
    pub lndep: R,
    /// The width dependence of `ndep`, `wndep`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `ndep` times meters; default 0.
    pub wndep: R,
    /// The area dependence of `ndep`, `pndep`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `ndep` times square
    /// meters; default 0.
    pub pndep: R,
    /// The source and drain doping concentration, `nsd`, in per cubic meter; default
    /// 1e+26.
    pub nsd: R,
    /// The length dependence of `nsd`, `lnsd`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `nsd` times meters; default 0.
    pub lnsd: R,
    /// The width dependence of `nsd`, `wnsd`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `nsd` times meters; default 0.
    pub wnsd: R,
    /// The area dependence of `nsd`, `pnsd`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `nsd` times square
    /// meters; default 0.
    pub pnsd: R,
    /// The adjusting parameter for surface potential due to non-uniform vertical
    /// doping, `phin`, in volts; default 0.
    pub phin: R,
    /// The length dependence of `phin`, `lphin`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `phin` times meters; default 0.
    pub lphin: R,
    /// The width dependence of `phin`, `wphin`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `phin` times meters; default 0.
    pub wphin: R,
    /// The area dependence of `phin`, `pphin`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `phin` times square
    /// meters; default 0.
    pub pphin: R,
    /// The poly-gate doping concentration, `ngate`, in per cubic meter; default 0.
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
    /// derived from `ndep`.
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
    /// default derived from `ndep` and `xt`.
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
    /// The depth of the doping near the surface, `xt`, in meters; default 1.55e-07.
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
    /// The temperature coefficient of the threshold, `kt1`, in volts; default -0.11.
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
    /// The channel-length dependence of the threshold's temperature coefficient,
    /// `kt1l`, in volt meters; default 0.
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
    /// The body-coefficient of kt1, `kt2`; default 0.022.
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
    /// The narrow width effect coefficient, `k3`; default 80.
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
    /// The body effect coefficient of k3, `k3b`, in per volt; default 0.
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
    /// The narrow width effect parameter, `w0`, in meters; default 2.5e-06.
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
    /// The first parameter for Vth shift due to pocket, `dvtp0`, in meters; default 0.
    pub dvtp0: R,
    /// The length dependence of `dvtp0`, `ldvtp0`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `dvtp0` times meters; default 0.
    pub ldvtp0: R,
    /// The width dependence of `dvtp0`, `wdvtp0`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `dvtp0` times meters; default 0.
    pub wdvtp0: R,
    /// The area dependence of `dvtp0`, `pdvtp0`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `dvtp0` times
    /// square meters; default 0.
    pub pdvtp0: R,
    /// The second parameter for Vth shift due to pocket, `dvtp1`, in per volt; default
    /// 0.
    pub dvtp1: R,
    /// The length dependence of `dvtp1`, `ldvtp1`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `dvtp1` times meters; default 0.
    pub ldvtp1: R,
    /// The width dependence of `dvtp1`, `wdvtp1`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `dvtp1` times meters; default 0.
    pub wdvtp1: R,
    /// The area dependence of `dvtp1`, `pdvtp1`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `dvtp1` times
    /// square meters; default 0.
    pub pdvtp1: R,
    /// The equivalent length of pocket region at zero bias, `lpe0`, in meters; default
    /// 1.74e-07.
    pub lpe0: R,
    /// The length dependence of `lpe0`, `llpe0`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `lpe0` times meters; default 0.
    pub llpe0: R,
    /// The width dependence of `lpe0`, `wlpe0`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `lpe0` times meters; default 0.
    pub wlpe0: R,
    /// The area dependence of `lpe0`, `plpe0`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `lpe0` times square
    /// meters; default 0.
    pub plpe0: R,
    /// The equivalent length of pocket region accounting for body bias, `lpeb`, in
    /// meters; default 0.
    pub lpeb: R,
    /// The length dependence of `lpeb`, `llpeb`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `lpeb` times meters; default 0.
    pub llpeb: R,
    /// The width dependence of `lpeb`, `wlpeb`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `lpeb` times meters; default 0.
    pub wlpeb: R,
    /// The area dependence of `lpeb`, `plpeb`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `lpeb` times square
    /// meters; default 0.
    pub plpeb: R,
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
    /// The body-bias coefficient of the short-channel effect, `dvt2`, in per volt;
    /// default -0.032.
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
    /// The first coefficient of the narrow-width effect, `dvt0w`, in per meter; default
    /// 0.
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
    /// The second coefficient of the narrow-width effect, `dvt1w`, in per meter;
    /// default 5.3e+06.
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
    /// The body-bias coefficient of the narrow-width effect, `dvt2w`, in per volt;
    /// default -0.032.
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
    /// The DIBL coefficient of output resistance, `drout`; default 0.56.
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
    /// The DIBL coefficient in the subthreshold region, `dsub`; default `drout`.
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
    /// The linear gate dependence of mobility, `ua`, in meters per volt; default 1e-9
    /// m/V, ngspice's at `mobmod` 0.
    pub ua: Option<R>,
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
    /// The temperature coefficient of ua, `ua1`, in meters per volt; default 1e-09.
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
    /// The quadratic gate dependence of mobility, `ub`, in square meters per square
    /// volt; default 1e-19.
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
    /// The temperature coefficient of ub, `ub1`, in square meters per square volt;
    /// default -1e-18.
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
    /// The body-bias dependence of mobility, `uc`, in meters per square volt; default
    /// -4.65e-11 m/V², ngspice's at `mobmod` 0.
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
    /// The temperature coefficient of uc, `uc1`, in meters per square volt; default
    /// -5.6e-11 m/V², ngspice's at `mobmod` 0.
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
    /// The coulomb scattering factor of mobility, `ud`, in per square meter; default
    /// 1e+14.
    pub ud: R,
    /// The length dependence of `ud`, `lud`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `ud` times meters; default 0.
    pub lud: R,
    /// The width dependence of `ud`, `wud`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `ud` times meters; default 0.
    pub wud: R,
    /// The area dependence of `ud`, `pud`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `ud` times square
    /// meters; default 0.
    pub pud: R,
    /// The temperature coefficient of ud, `ud1`, in per square meter; default 0.
    pub ud1: R,
    /// The length dependence of `ud1`, `lud1`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `ud1` times meters; default 0.
    pub lud1: R,
    /// The width dependence of `ud1`, `wud1`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `ud1` times meters; default 0.
    pub wud1: R,
    /// The area dependence of `ud1`, `pud1`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `ud1` times square
    /// meters; default 0.
    pub pud1: R,
    /// The channel length linear factor of mobility, `up`, in per square meter; default
    /// 0.
    pub up: R,
    /// The length dependence of `up`, `lup`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `up` times meters; default 0.
    pub lup: R,
    /// The width dependence of `up`, `wup`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `up` times meters; default 0.
    pub wup: R,
    /// The area dependence of `up`, `pup`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `up` times square
    /// meters; default 0.
    pub pup: R,
    /// The channel length exponential factor of mobility, `lp`, in meters; default
    /// 1e-08.
    pub lp: R,
    /// The length dependence of `lp`, `llp`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `lp` times meters; default 0.
    pub llp: R,
    /// The width dependence of `lp`, `wlp`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `lp` times meters; default 0.
    pub wlp: R,
    /// The area dependence of `lp`, `plp`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `lp` times square
    /// meters; default 0.
    pub plp: R,
    /// The low-field mobility at Tnom, `u0`, in square meters per volt-second; default
    /// 0.067 m²/(V·s) for an n-channel and 0.025 m²/(V·s) for a p-channel.
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
    /// The mobility exponent, `eu`; default 1.67 for an n-channel and 1 for a
    /// p-channel.
    pub eu: Option<R>,
    /// The length dependence of `eu`, `leu`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `eu` times meters; default 0.
    pub leu: R,
    /// The width dependence of `eu`, `weu`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `eu` times meters; default 0.
    pub weu: R,
    /// The area dependence of `eu`, `peu`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `eu` times square
    /// meters; default 0.
    pub peu: R,
    /// The temperature coefficient of mobility, `ute`; default -1.5.
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
    /// The threshold voltage offset, `voff`, in volts; default -0.08.
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
    /// The fitting parameter of moderate inversion in Vgsteff, `minv`; default 0.
    pub minv: R,
    /// The length dependence of `minv`, `lminv`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `minv` times meters; default 0.
    pub lminv: R,
    /// The width dependence of `minv`, `wminv`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `minv` times meters; default 0.
    pub wminv: R,
    /// The area dependence of `minv`, `pminv`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `minv` times square
    /// meters; default 0.
    pub pminv: R,
    /// The channel-length dependence of the threshold offset, `voffl`, in volt meters;
    /// default 0.
    pub voffl: R,
    /// The gate-source overlap capacitance per width, `cgso`, in farads per meter;
    /// default derived from `dlc` and `cgsl`, or from `xj`.
    pub cgso: Option<R>,
    /// The gate-drain overlap capacitance per width, `cgdo`, in farads per meter;
    /// default derived from `dlc` and `cgdl`, or from `xj`.
    pub cgdo: Option<R>,
    /// The gate-bulk overlap capacitance per length, `cgbo`, in farads per meter;
    /// default twice `dwc` times the oxide capacitance.
    pub cgbo: Option<R>,
    /// The effective Vds parameter, `delta`, in volts; default 0.01.
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
    /// The sheet resistance of the source and drain diffusions, `rsh`, in ohms; default
    /// 0.
    pub rsh: R,
    /// The source and drain resistance per width, `rdsw`, in ohms; default 200.
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
    /// The source and drain resistance per width at high gate bias, `rdswmin`, in ohms;
    /// default 0.
    pub rdswmin: R,
    /// The source resistance per width, `rsw`, in ohms; default 100.
    pub rsw: R,
    /// The length dependence of `rsw`, `lrsw`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `rsw` times meters; default 0.
    pub lrsw: R,
    /// The width dependence of `rsw`, `wrsw`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `rsw` times meters; default 0.
    pub wrsw: R,
    /// The area dependence of `rsw`, `prsw`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `rsw` times square
    /// meters; default 0.
    pub prsw: R,
    /// The drain resistance per width, `rdw`, in ohms; default 100.
    pub rdw: R,
    /// The length dependence of `rdw`, `lrdw`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `rdw` times meters; default 0.
    pub lrdw: R,
    /// The width dependence of `rdw`, `wrdw`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `rdw` times meters; default 0.
    pub wrdw: R,
    /// The area dependence of `rdw`, `prdw`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `rdw` times square
    /// meters; default 0.
    pub prdw: R,
    /// The drain resistance per width at high Vg, `rdwmin`, in ohms; default 0.
    pub rdwmin: R,
    /// The source resistance per width at high Vg, `rswmin`, in ohms; default 0.
    pub rswmin: R,
    /// The gate-bias effect on parasitic resistance, `prwg`, in per volt; default 1.
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
    /// The body-effect on parasitic resistance, `prwb`, in per square-root volt;
    /// default 0.
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
    /// The temperature coefficient of parasitic resistance, `prt`, in ohms; default 0.
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
    /// The drain-induced barrier lowering in the subthreshold region, `eta0`; default
    /// 0.08.
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
    /// The body-bias coefficient of `eta0`, `etab`, in per volt; default -0.07.
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
    /// The channel length modulation Coefficient, `pclm`; default 1.3.
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
    /// The first coefficient of the drain-induced barrier lowering of the output
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
    /// The second coefficient of the drain-induced barrier lowering of the output
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
    /// The body-effect on drain-induced barrier lowering, `pdiblcb`, in per volt;
    /// default 0.
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
    /// The rout degradation coefficient for pocket devices, `fprout`, in volts per
    /// square-root meter; default 0.
    pub fprout: R,
    /// The length dependence of `fprout`, `lfprout`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `fprout` times meters; default 0.
    pub lfprout: R,
    /// The width dependence of `fprout`, `wfprout`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `fprout` times meters; default 0.
    pub wfprout: R,
    /// The area dependence of `fprout`, `pfprout`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `fprout` times
    /// square meters; default 0.
    pub pfprout: R,
    /// The coefficient for drain-induced Vth shifts, `pdits`, in per volt; default 0.
    pub pdits: R,
    /// The length dependence of `pdits`, `lpdits`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `pdits` times meters; default 0.
    pub lpdits: R,
    /// The width dependence of `pdits`, `wpdits`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `pdits` times meters; default 0.
    pub wpdits: R,
    /// The area dependence of `pdits`, `ppdits`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `pdits` times
    /// square meters; default 0.
    pub ppdits: R,
    /// The length dependence of drain-induced Vth shifts, `pditsl`, in per meter;
    /// default 0.
    pub pditsl: R,
    /// The vds dependence of drain-induced Vth shifts, `pditsd`, in per volt; default
    /// 0.
    pub pditsd: R,
    /// The length dependence of `pditsd`, `lpditsd`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `pditsd` times meters; default 0.
    pub lpditsd: R,
    /// The width dependence of `pditsd`, `wpditsd`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `pditsd` times meters; default 0.
    pub wpditsd: R,
    /// The area dependence of `pditsd`, `ppditsd`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `pditsd` times
    /// square meters; default 0.
    pub ppditsd: R,
    /// The exponential coefficient of the substrate current's effect on the output
    /// resistance, `pscbe1`, in volts per meter; default 4.24e+08.
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
    /// The coefficient of the substrate current's effect on the output resistance,
    /// `pscbe2`, in meters per volt; default 1e-05.
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
    /// The gate dependence of output resistance parameter, `pvag`; default 0.
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
    /// The bottom source junction reverse saturation current density, `jss`, in amperes
    /// per square meter; default 0.0001.
    pub jss: R,
    /// The isolation edge sidewall source junction reverse saturation current density,
    /// `jsws`, in amperes per meter; default 0.
    pub jsws: R,
    /// The gate edge source junction reverse saturation current density, `jswgs`, in
    /// amperes per meter; default 0.
    pub jswgs: R,
    /// The source junction built-in potential, `pbs`, in volts; default 1.
    pub pbs: R,
    /// The source junction emission coefficient, `njs`; default 1.
    pub njs: R,
    /// The source junction current temperature exponent, `xtis`; default 3.
    pub xtis: R,
    /// The source bottom junction capacitance grading coefficient, `mjs`; default 0.5.
    pub mjs: R,
    /// The source sidewall junction capacitance built in potential, `pbsws`, in volts;
    /// default 1.
    pub pbsws: R,
    /// The source sidewall junction capacitance grading coefficient, `mjsws`; default
    /// 0.33.
    pub mjsws: R,
    /// The source (gate side) sidewall junction capacitance built in potential,
    /// `pbswgs`, in volts; default `pbsws`.
    pub pbswgs: Option<R>,
    /// The source (gate side) sidewall junction capacitance grading coefficient,
    /// `mjswgs`; default `mjsws`.
    pub mjswgs: Option<R>,
    /// The source bottom junction capacitance per unit area, `cjs`, in farads per
    /// square meter; default 0.0005.
    pub cjs: R,
    /// The source sidewall junction capacitance per unit periphery, `cjsws`, in farads
    /// per meter; default 5e-10.
    pub cjsws: R,
    /// The source (gate side) sidewall junction capacitance per unit width, `cjswgs`,
    /// in farads per meter; default `cjsws`.
    pub cjswgs: Option<R>,
    /// The bottom drain junction reverse saturation current density, `jsd`, in amperes
    /// per square meter; default `jss`.
    pub jsd: Option<R>,
    /// The isolation edge sidewall drain junction reverse saturation current density,
    /// `jswd`, in amperes per meter; default `jsws`.
    pub jswd: Option<R>,
    /// The gate edge drain junction reverse saturation current density, `jswgd`, in
    /// amperes per meter; default `jswgs`.
    pub jswgd: Option<R>,
    /// The drain junction built-in potential, `pbd`, in volts; default `pbs`.
    pub pbd: Option<R>,
    /// The drain junction emission coefficient, `njd`; default `njs`.
    pub njd: Option<R>,
    /// The drain junction current temperature exponent, `xtid`; default `xtis`.
    pub xtid: Option<R>,
    /// The drain bottom junction capacitance grading coefficient, `mjd`; default `mjs`.
    pub mjd: Option<R>,
    /// The drain sidewall junction capacitance built in potential, `pbswd`, in volts;
    /// default `pbsws`.
    pub pbswd: Option<R>,
    /// The drain sidewall junction capacitance grading coefficient, `mjswd`; default
    /// `mjsws`.
    pub mjswd: Option<R>,
    /// The drain (gate side) sidewall junction capacitance built in potential,
    /// `pbswgd`, in volts; default `pbswgs`.
    pub pbswgd: Option<R>,
    /// The drain (gate side) sidewall junction capacitance grading coefficient,
    /// `mjswgd`; default `mjswgs`.
    pub mjswgd: Option<R>,
    /// The drain bottom junction capacitance per unit area, `cjd`, in farads per square
    /// meter; default `cjs`.
    pub cjd: Option<R>,
    /// The drain sidewall junction capacitance per unit periphery, `cjswd`, in farads
    /// per meter; default `cjsws`.
    pub cjswd: Option<R>,
    /// The drain (gate side) sidewall junction capacitance per unit width, `cjswgd`, in
    /// farads per meter; default `cjswgs`.
    pub cjswgd: Option<R>,
    /// The flat-band voltage of the charges at `capmod` 0, `vfbcv`, in volts; default
    /// -1.
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
    /// The temperature coefficient of pb, `tpb`, in volts per kelvin; default 0.
    pub tpb: R,
    /// The temperature coefficient of cj, `tcj`, in per kelvin; default 0.
    pub tcj: R,
    /// The temperature coefficient of pbsw, `tpbsw`, in volts per kelvin; default 0.
    pub tpbsw: R,
    /// The temperature coefficient of cjsw, `tcjsw`, in per kelvin; default 0.
    pub tcjsw: R,
    /// The temperature coefficient of pbswg, `tpbswg`, in volts per kelvin; default 0.
    pub tpbswg: R,
    /// The temperature coefficient of cjswg, `tcjswg`, in per kelvin; default 0.
    pub tcjswg: R,
    /// The exponential coefficient for finite charge thickness, `acde`, in meters per
    /// volt; default 1.
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
    /// The coefficient for gate-bias dependent surface potential, `moin`; default 15.
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
    /// The C-V turn-on/off parameter, `noff`; default 1.
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
    /// The C-V lateral-shift parameter, `voffcv`, in volts; default 0.
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
    /// The distance of Mid-Contact to Gate edge, `dmcg`, in meters; default 0.
    pub dmcg: R,
    /// The distance of Mid-Contact to Isolation, `dmci`, in meters; default `dmcg`.
    pub dmci: Option<R>,
    /// The distance of Mid-Diffusion to Gate edge, `dmdg`, in meters; default 0.
    pub dmdg: R,
    /// The distance of Mid-Contact to Gate edge in Test structures, `dmcgt`, in meters;
    /// default 0.
    pub dmcgt: R,
    /// The distance from gate contact center to device edge, `xgw`, in meters; default
    /// 0.
    pub xgw: R,
    /// The offset of the gate's drawn length, `xgl`, in meters; default 0.
    pub xgl: R,
    /// The gate sheet resistance, `rshg`, in ohms; default 0.1.
    pub rshg: R,
    /// The number of gate contacts, `ngcon`; default 1.
    pub ngcon: R,
    /// The first fitting parameter of the bias-dependent gate resistance, `xrcrg1`;
    /// default 12.
    pub xrcrg1: R,
    /// The length dependence of `xrcrg1`, `lxrcrg1`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `xrcrg1` times meters; default 0.
    pub lxrcrg1: R,
    /// The width dependence of `xrcrg1`, `wxrcrg1`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `xrcrg1` times meters; default 0.
    pub wxrcrg1: R,
    /// The area dependence of `xrcrg1`, `pxrcrg1`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `xrcrg1` times
    /// square meters; default 0.
    pub pxrcrg1: R,
    /// The second fitting parameter of the bias-dependent gate resistance, `xrcrg2`;
    /// default 1.
    pub xrcrg2: R,
    /// The length dependence of `xrcrg2`, `lxrcrg2`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `xrcrg2` times meters; default 0.
    pub lxrcrg2: R,
    /// The width dependence of `xrcrg2`, `wxrcrg2`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `xrcrg2` times meters; default 0.
    pub wxrcrg2: R,
    /// The area dependence of `xrcrg2`, `pxrcrg2`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `xrcrg2` times
    /// square meters; default 0.
    pub pxrcrg2: R,
    /// The velocity overshoot parameter, `lambda`; default 0, and no velocity overshoot
    /// where it is not given.
    pub lambda: Option<R>,
    /// The length dependence of `lambda`, `llambda`: its coefficient of the reciprocal
    /// of the effective length, in the unit of `lambda` times meters; default 0.
    pub llambda: R,
    /// The width dependence of `lambda`, `wlambda`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `lambda` times meters; default 0.
    pub wlambda: R,
    /// The area dependence of `lambda`, `plambda`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `lambda` times
    /// square meters; default 0.
    pub plambda: R,
    /// The thermal velocity, `vtl`, in meters per second; default 2e5 m/s, and no limit
    /// of the source end's velocity where it is not given.
    pub vtl: Option<R>,
    /// The length dependence of `vtl`, `lvtl`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `vtl` times meters; default 0.
    pub lvtl: R,
    /// The width dependence of `vtl`, `wvtl`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `vtl` times meters; default 0.
    pub wvtl: R,
    /// The area dependence of `vtl`, `pvtl`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `vtl` times square
    /// meters; default 0.
    pub pvtl: R,
    /// The backscattering mean free path, `lc`, in meters; default 5e-09.
    pub lc: R,
    /// The backscattering coefficient, `xn`; default 3.
    pub xn: R,
    /// The length dependence of `xn`, `lxn`: its coefficient of the reciprocal of the
    /// effective length, in the unit of `xn` times meters; default 0.
    pub lxn: R,
    /// The width dependence of `xn`, `wxn`: its coefficient of the reciprocal of the
    /// effective width, in the unit of `xn` times meters; default 0.
    pub wxn: R,
    /// The area dependence of `xn`, `pxn`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `xn` times square
    /// meters; default 0.
    pub pxn: R,
    /// The S/D flatband voltage offset, `vfbsdoff`, in volts; default 0.
    pub vfbsdoff: R,
    /// The length dependence of `vfbsdoff`, `lvfbsdoff`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `vfbsdoff` times meters;
    /// default 0.
    pub lvfbsdoff: R,
    /// The width dependence of `vfbsdoff`, `wvfbsdoff`: its coefficient of the
    /// reciprocal of the effective width, in the unit of `vfbsdoff` times meters;
    /// default 0.
    pub wvfbsdoff: R,
    /// The area dependence of `vfbsdoff`, `pvfbsdoff`: its coefficient of the
    /// reciprocal of the effective length times the effective width, in the unit of
    /// `vfbsdoff` times square meters; default 0.
    pub pvfbsdoff: R,
    /// The temperature parameter for vfbsdoff, `tvfbsdoff`, in per kelvin; default 0.
    pub tvfbsdoff: R,
    /// The length dependence of `tvfbsdoff`, `ltvfbsdoff`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `tvfbsdoff` times meters;
    /// default 0.
    pub ltvfbsdoff: R,
    /// The width dependence of `tvfbsdoff`, `wtvfbsdoff`: its coefficient of the
    /// reciprocal of the effective width, in the unit of `tvfbsdoff` times meters;
    /// default 0.
    pub wtvfbsdoff: R,
    /// The area dependence of `tvfbsdoff`, `ptvfbsdoff`: its coefficient of the
    /// reciprocal of the effective length times the effective width, in the unit of
    /// `tvfbsdoff` times square meters; default 0.
    pub ptvfbsdoff: R,
    /// The temperature parameter for voff, `tvoff`, in per kelvin; default 0.
    pub tvoff: R,
    /// The length dependence of `tvoff`, `ltvoff`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `tvoff` times meters; default 0.
    pub ltvoff: R,
    /// The width dependence of `tvoff`, `wtvoff`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `tvoff` times meters; default 0.
    pub wtvoff: R,
    /// The area dependence of `tvoff`, `ptvoff`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `tvoff` times
    /// square meters; default 0.
    pub ptvoff: R,
    /// The length each side of the channel loses in the flicker noise, beyond `lint`,
    /// `lintnoi`, in meters; default 0.
    pub lintnoi: R,
    /// The length lost on each side of the channel, `lint`, in meters; default 0.
    pub lint: R,
    /// The coefficient of the length dependence of the length lost, `ll`, in the
    /// manual's units; default 0.
    pub ll: R,
    /// The coefficient of the length dependence of the length lost in the charges,
    /// `llc`, in the manual's units; default `ll`.
    pub llc: Option<R>,
    /// The exponent of the length dependence of the length lost, `lln`; default 1.
    pub lln: R,
    /// The coefficient of the width dependence of the length lost, `lw`, in the
    /// manual's units; default 0.
    pub lw: R,
    /// The coefficient of the width dependence of the length lost in the charges,
    /// `lwc`, in the manual's units; default `lw`.
    pub lwc: Option<R>,
    /// The exponent of the width dependence of the length lost, `lwn`; default 1.
    pub lwn: R,
    /// The coefficient of the area dependence of the length lost, `lwl`, in the
    /// manual's units; default 0.
    pub lwl: R,
    /// The coefficient of the area dependence of the length lost in the charges,
    /// `lwlc`, in the manual's units; default `lwl`.
    pub lwlc: Option<R>,
    /// The width exponent of the source and drain resistance, `wr`; default 1.
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
    /// The width lost on each side of the channel, `wint`, in meters; default 0.
    pub wint: R,
    /// The gate-bias dependence of the width lost, `dwg`, in meters per volt; default
    /// 0.
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
    /// The body-bias dependence of the width lost, `dwb`, in meters per square-root
    /// volt; default 0.
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
    /// The coefficient of the length dependence of the width lost, `wl`, in the
    /// manual's units; default 0.
    pub wl: R,
    /// The coefficient of the length dependence of the width lost in the charges,
    /// `wlc`, in the manual's units; default `wl`.
    pub wlc: Option<R>,
    /// The exponent of the length dependence of the width lost, `wln`; default 1.
    pub wln: R,
    /// The coefficient of the width dependence of the width lost, `ww`, in the manual's
    /// units; default 0.
    pub ww: R,
    /// The coefficient of the width dependence of the width lost in the charges, `wwc`,
    /// in the manual's units; default `ww`.
    pub wwc: Option<R>,
    /// The exponent of the width dependence of the width lost, `wwn`; default 1.
    pub wwn: R,
    /// The coefficient of the area dependence of the width lost, `wwl`, in the manual's
    /// units; default 0.
    pub wwl: R,
    /// The coefficient of the area dependence of the width lost in the charges, `wwlc`,
    /// in the manual's units; default `wwl`.
    pub wwlc: Option<R>,
    /// The first narrow-width coefficient of Abulk, `b0`, in meters; default 0.
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
    /// The second narrow-width coefficient of Abulk, `b1`, in meters; default 0.
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
    /// The gate-source overlap capacitance per width of the lightly doped extension,
    /// `cgsl`, in farads per meter; default 0.
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
    /// The gate-drain overlap capacitance per width of the lightly doped extension,
    /// `cgdl`, in farads per meter; default 0.
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
    /// The coefficient of the bias dependence of the gate-source overlap, `ckappas`, in
    /// volts; default 0.6.
    pub ckappas: R,
    /// The length dependence of `ckappas`, `lckappas`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `ckappas` times meters;
    /// default 0.
    pub lckappas: R,
    /// The width dependence of `ckappas`, `wckappas`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `ckappas` times meters; default 0.
    pub wckappas: R,
    /// The area dependence of `ckappas`, `pckappas`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `ckappas`
    /// times square meters; default 0.
    pub pckappas: R,
    /// The coefficient of the bias dependence of the gate-drain overlap, `ckappad`, in
    /// volts; default `ckappas`.
    pub ckappad: Option<R>,
    /// The length dependence of `ckappad`, `lckappad`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `ckappad` times meters;
    /// default 0.
    pub lckappad: R,
    /// The width dependence of `ckappad`, `wckappad`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `ckappad` times meters; default 0.
    pub wckappad: R,
    /// The area dependence of `ckappad`, `pckappad`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `ckappad`
    /// times square meters; default 0.
    pub pckappad: R,
    /// The fringe capacitance parameter, `cf`, in farads per meter; default `2 · epsrox
    /// · ε0 / π · ln(1 + 0.4 µm / toxe)`, of `toxe` where it is given, else of 3 nm.
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
    /// The vdsat parameter for C-V model, `clc`, in meters; default 1e-07.
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
    /// The vdsat parameter for C-V model, `cle`; default 0.6.
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
    /// The width lost on each side of the channel in the charges, `dwc`, in meters;
    /// default `wint`.
    pub dwc: Option<R>,
    /// The length lost on each side of the channel in the charges, `dlc`, in meters;
    /// default `lint`.
    pub dlc: Option<R>,
    /// The offset of the channel's width from masking and etching, `xw`, in meters;
    /// default 0.
    pub xw: R,
    /// The offset of the channel's length from masking and etching, `xl`, in meters;
    /// default 0.
    pub xl: R,
    /// The length lost on each side of the channel in the gate tunneling currents,
    /// `dlcig`, in meters; default `lint`.
    pub dlcig: Option<R>,
    /// The width lost on each side of the source and drain junctions, `dwj`, in meters;
    /// default `dwc`.
    pub dwj: Option<R>,
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
    /// The channel-length dependence of the substrate current, `alpha1`, in per volt;
    /// default 0.
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
    /// The exponential coefficient of the substrate current, `beta0`, in volts; default
    /// 0.
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
    /// The pre-exponential constant for GIDL, `agidl`, in siemens; default 0.
    pub agidl: R,
    /// The length dependence of `agidl`, `lagidl`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `agidl` times meters; default 0.
    pub lagidl: R,
    /// The width dependence of `agidl`, `wagidl`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `agidl` times meters; default 0.
    pub wagidl: R,
    /// The area dependence of `agidl`, `pagidl`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `agidl` times
    /// square meters; default 0.
    pub pagidl: R,
    /// The exponential constant for GIDL, `bgidl`, in volts per meter; default 2.3e+09.
    pub bgidl: R,
    /// The length dependence of `bgidl`, `lbgidl`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `bgidl` times meters; default 0.
    pub lbgidl: R,
    /// The width dependence of `bgidl`, `wbgidl`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `bgidl` times meters; default 0.
    pub wbgidl: R,
    /// The area dependence of `bgidl`, `pbgidl`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `bgidl` times
    /// square meters; default 0.
    pub pbgidl: R,
    /// The parameter for body-bias dependence of GIDL, `cgidl`, in cubic volts; default
    /// 0.5.
    pub cgidl: R,
    /// The length dependence of `cgidl`, `lcgidl`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `cgidl` times meters; default 0.
    pub lcgidl: R,
    /// The width dependence of `cgidl`, `wcgidl`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `cgidl` times meters; default 0.
    pub wcgidl: R,
    /// The area dependence of `cgidl`, `pcgidl`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `cgidl` times
    /// square meters; default 0.
    pub pcgidl: R,
    /// The band bending of the gate-induced drain leakage, `egidl`, in volts; default
    /// 0.8.
    pub egidl: R,
    /// The length dependence of `egidl`, `legidl`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `egidl` times meters; default 0.
    pub legidl: R,
    /// The width dependence of `egidl`, `wegidl`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `egidl` times meters; default 0.
    pub wegidl: R,
    /// The area dependence of `egidl`, `pegidl`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `egidl` times
    /// square meters; default 0.
    pub pegidl: R,
    /// The parameter for Igc, `aigc`, in the manual's units; default 1.36e-2 for an
    /// n-channel and 9.8e-3 for a p-channel.
    pub aigc: Option<R>,
    /// The length dependence of `aigc`, `laigc`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `aigc` times meters; default 0.
    pub laigc: R,
    /// The width dependence of `aigc`, `waigc`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `aigc` times meters; default 0.
    pub waigc: R,
    /// The area dependence of `aigc`, `paigc`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `aigc` times square
    /// meters; default 0.
    pub paigc: R,
    /// The parameter for Igc, `bigc`, in the manual's units; default 1.71e-3 for an
    /// n-channel and 7.59e-4 for a p-channel.
    pub bigc: Option<R>,
    /// The length dependence of `bigc`, `lbigc`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `bigc` times meters; default 0.
    pub lbigc: R,
    /// The width dependence of `bigc`, `wbigc`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `bigc` times meters; default 0.
    pub wbigc: R,
    /// The area dependence of `bigc`, `pbigc`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `bigc` times square
    /// meters; default 0.
    pub pbigc: R,
    /// The parameter for Igc, `cigc`, in per volt; default 0.075 per volt for an
    /// n-channel and 0.03 per volt for a p-channel.
    pub cigc: Option<R>,
    /// The length dependence of `cigc`, `lcigc`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `cigc` times meters; default 0.
    pub lcigc: R,
    /// The width dependence of `cigc`, `wcigc`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `cigc` times meters; default 0.
    pub wcigc: R,
    /// The area dependence of `cigc`, `pcigc`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `cigc` times square
    /// meters; default 0.
    pub pcigc: R,
    /// The parameter for Igs,d, `aigsd`, in the manual's units; default 1.36e-2 for an
    /// n-channel and 9.8e-3 for a p-channel.
    pub aigsd: Option<R>,
    /// The length dependence of `aigsd`, `laigsd`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `aigsd` times meters; default 0.
    pub laigsd: R,
    /// The width dependence of `aigsd`, `waigsd`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `aigsd` times meters; default 0.
    pub waigsd: R,
    /// The area dependence of `aigsd`, `paigsd`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `aigsd` times
    /// square meters; default 0.
    pub paigsd: R,
    /// The parameter for Igs,d, `bigsd`, in the manual's units; default 1.71e-3 for an
    /// n-channel and 7.59e-4 for a p-channel.
    pub bigsd: Option<R>,
    /// The length dependence of `bigsd`, `lbigsd`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `bigsd` times meters; default 0.
    pub lbigsd: R,
    /// The width dependence of `bigsd`, `wbigsd`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `bigsd` times meters; default 0.
    pub wbigsd: R,
    /// The area dependence of `bigsd`, `pbigsd`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `bigsd` times
    /// square meters; default 0.
    pub pbigsd: R,
    /// The parameter for Igs,d, `cigsd`, in per volt; default 0.075 per volt for an
    /// n-channel and 0.03 per volt for a p-channel.
    pub cigsd: Option<R>,
    /// The length dependence of `cigsd`, `lcigsd`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `cigsd` times meters; default 0.
    pub lcigsd: R,
    /// The width dependence of `cigsd`, `wcigsd`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `cigsd` times meters; default 0.
    pub wcigsd: R,
    /// The area dependence of `cigsd`, `pcigsd`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `cigsd` times
    /// square meters; default 0.
    pub pcigsd: R,
    /// The parameter for Igb, `aigbacc`, in the manual's units; default 0.0136.
    pub aigbacc: R,
    /// The length dependence of `aigbacc`, `laigbacc`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `aigbacc` times meters;
    /// default 0.
    pub laigbacc: R,
    /// The width dependence of `aigbacc`, `waigbacc`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `aigbacc` times meters; default 0.
    pub waigbacc: R,
    /// The area dependence of `aigbacc`, `paigbacc`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `aigbacc`
    /// times square meters; default 0.
    pub paigbacc: R,
    /// The parameter for Igb, `bigbacc`, in the manual's units; default 0.00171.
    pub bigbacc: R,
    /// The length dependence of `bigbacc`, `lbigbacc`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `bigbacc` times meters;
    /// default 0.
    pub lbigbacc: R,
    /// The width dependence of `bigbacc`, `wbigbacc`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `bigbacc` times meters; default 0.
    pub wbigbacc: R,
    /// The area dependence of `bigbacc`, `pbigbacc`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `bigbacc`
    /// times square meters; default 0.
    pub pbigbacc: R,
    /// The parameter for Igb, `cigbacc`, in per volt; default 0.075.
    pub cigbacc: R,
    /// The length dependence of `cigbacc`, `lcigbacc`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `cigbacc` times meters;
    /// default 0.
    pub lcigbacc: R,
    /// The width dependence of `cigbacc`, `wcigbacc`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `cigbacc` times meters; default 0.
    pub wcigbacc: R,
    /// The area dependence of `cigbacc`, `pcigbacc`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `cigbacc`
    /// times square meters; default 0.
    pub pcigbacc: R,
    /// The parameter for Igb, `aigbinv`, in the manual's units; default 0.0111.
    pub aigbinv: R,
    /// The length dependence of `aigbinv`, `laigbinv`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `aigbinv` times meters;
    /// default 0.
    pub laigbinv: R,
    /// The width dependence of `aigbinv`, `waigbinv`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `aigbinv` times meters; default 0.
    pub waigbinv: R,
    /// The area dependence of `aigbinv`, `paigbinv`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `aigbinv`
    /// times square meters; default 0.
    pub paigbinv: R,
    /// The parameter for Igb, `bigbinv`, in the manual's units; default 0.000949.
    pub bigbinv: R,
    /// The length dependence of `bigbinv`, `lbigbinv`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `bigbinv` times meters;
    /// default 0.
    pub lbigbinv: R,
    /// The width dependence of `bigbinv`, `wbigbinv`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `bigbinv` times meters; default 0.
    pub wbigbinv: R,
    /// The area dependence of `bigbinv`, `pbigbinv`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `bigbinv`
    /// times square meters; default 0.
    pub pbigbinv: R,
    /// The parameter for Igb, `cigbinv`, in per volt; default 0.006.
    pub cigbinv: R,
    /// The length dependence of `cigbinv`, `lcigbinv`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `cigbinv` times meters;
    /// default 0.
    pub lcigbinv: R,
    /// The width dependence of `cigbinv`, `wcigbinv`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `cigbinv` times meters; default 0.
    pub wcigbinv: R,
    /// The area dependence of `cigbinv`, `pcigbinv`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `cigbinv`
    /// times square meters; default 0.
    pub pcigbinv: R,
    /// The parameter for Igc slope, `nigc`; default 1.
    pub nigc: R,
    /// The length dependence of `nigc`, `lnigc`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `nigc` times meters; default 0.
    pub lnigc: R,
    /// The width dependence of `nigc`, `wnigc`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `nigc` times meters; default 0.
    pub wnigc: R,
    /// The area dependence of `nigc`, `pnigc`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `nigc` times square
    /// meters; default 0.
    pub pnigc: R,
    /// The parameter for Igbinv slope, `nigbinv`; default 3.
    pub nigbinv: R,
    /// The length dependence of `nigbinv`, `lnigbinv`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `nigbinv` times meters;
    /// default 0.
    pub lnigbinv: R,
    /// The width dependence of `nigbinv`, `wnigbinv`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `nigbinv` times meters; default 0.
    pub wnigbinv: R,
    /// The area dependence of `nigbinv`, `pnigbinv`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `nigbinv`
    /// times square meters; default 0.
    pub pnigbinv: R,
    /// The parameter for Igbacc slope, `nigbacc`; default 1.
    pub nigbacc: R,
    /// The length dependence of `nigbacc`, `lnigbacc`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `nigbacc` times meters;
    /// default 0.
    pub lnigbacc: R,
    /// The width dependence of `nigbacc`, `wnigbacc`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `nigbacc` times meters; default 0.
    pub wnigbacc: R,
    /// The area dependence of `nigbacc`, `pnigbacc`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `nigbacc`
    /// times square meters; default 0.
    pub pnigbacc: R,
    /// The exponent for Tox ratio, `ntox`; default 1.
    pub ntox: R,
    /// The length dependence of `ntox`, `lntox`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `ntox` times meters; default 0.
    pub lntox: R,
    /// The width dependence of `ntox`, `wntox`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `ntox` times meters; default 0.
    pub wntox: R,
    /// The area dependence of `ntox`, `pntox`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `ntox` times square
    /// meters; default 0.
    pub pntox: R,
    /// The parameter for the Si bandgap for Igbinv, `eigbinv`, in volts; default 1.1.
    pub eigbinv: R,
    /// The length dependence of `eigbinv`, `leigbinv`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `eigbinv` times meters;
    /// default 0.
    pub leigbinv: R,
    /// The width dependence of `eigbinv`, `weigbinv`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `eigbinv` times meters; default 0.
    pub weigbinv: R,
    /// The area dependence of `eigbinv`, `peigbinv`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `eigbinv`
    /// times square meters; default 0.
    pub peigbinv: R,
    /// The parameter for Igc partition, `pigcd`; default 1.
    pub pigcd: R,
    /// The length dependence of `pigcd`, `lpigcd`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `pigcd` times meters; default 0.
    pub lpigcd: R,
    /// The width dependence of `pigcd`, `wpigcd`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `pigcd` times meters; default 0.
    pub wpigcd: R,
    /// The area dependence of `pigcd`, `ppigcd`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `pigcd` times
    /// square meters; default 0.
    pub ppigcd: R,
    /// The factor for the gate edge Tox, `poxedge`; default 1.
    pub poxedge: R,
    /// The length dependence of `poxedge`, `lpoxedge`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `poxedge` times meters;
    /// default 0.
    pub lpoxedge: R,
    /// The width dependence of `poxedge`, `wpoxedge`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `poxedge` times meters; default 0.
    pub wpoxedge: R,
    /// The area dependence of `poxedge`, `ppoxedge`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `poxedge`
    /// times square meters; default 0.
    pub ppoxedge: R,
    /// The forward drain diode forward limiting current, `ijthdfwd`, in amperes;
    /// default `ijthsfwd`.
    pub ijthdfwd: Option<R>,
    /// The forward source diode forward limiting current, `ijthsfwd`, in amperes;
    /// default 0.1.
    pub ijthsfwd: R,
    /// The reverse drain diode forward limiting current, `ijthdrev`, in amperes;
    /// default `ijthsrev`.
    pub ijthdrev: Option<R>,
    /// The reverse source diode forward limiting current, `ijthsrev`, in amperes;
    /// default 0.1.
    pub ijthsrev: R,
    /// The fitting parameter for drain diode breakdown current, `xjbvd`; default
    /// `xjbvs`.
    pub xjbvd: Option<R>,
    /// The fitting parameter for source diode breakdown current, `xjbvs`; default 1.
    pub xjbvs: R,
    /// The drain diode breakdown voltage, `bvd`, in volts; default `bvs`.
    pub bvd: Option<R>,
    /// The source diode breakdown voltage, `bvs`, in volts; default 10.
    pub bvs: R,
    /// The source bottom trap-assisted saturation current density, `jtss`, in amperes
    /// per square meter; default 0.
    pub jtss: R,
    /// The drain bottom trap-assisted saturation current density, `jtsd`, in amperes
    /// per square meter; default `jtss`.
    pub jtsd: Option<R>,
    /// The source STI sidewall trap-assisted saturation current density, `jtssws`, in
    /// amperes per meter; default 0.
    pub jtssws: R,
    /// The drain STI sidewall trap-assisted saturation current density, `jtsswd`, in
    /// amperes per meter; default `jtssws`.
    pub jtsswd: Option<R>,
    /// The source gate-edge sidewall trap-assisted saturation current density,
    /// `jtsswgs`, in amperes per meter; default 0.
    pub jtsswgs: R,
    /// The drain gate-edge sidewall trap-assisted saturation current density,
    /// `jtsswgd`, in amperes per meter; default `jtsswgs`.
    pub jtsswgd: Option<R>,
    /// The non-ideality factor for bottom junction, `njts`; default 20.
    pub njts: R,
    /// The non-ideality factor for STI sidewall junction, `njtssw`; default 20.
    pub njtssw: R,
    /// The non-ideality factor for gate-edge sidewall junction, `njtsswg`; default 20.
    pub njtsswg: R,
    /// The power dependence of JTSS on temperature, `xtss`; default 0.02.
    pub xtss: R,
    /// The power dependence of JTSD on temperature, `xtsd`; default `xtss`.
    pub xtsd: Option<R>,
    /// The power dependence of JTSSWS on temperature, `xtssws`; default 0.02.
    pub xtssws: R,
    /// The power dependence of JTSSWD on temperature, `xtsswd`; default `xtssws`.
    pub xtsswd: Option<R>,
    /// The power dependence of JTSSWGS on temperature, `xtsswgs`; default 0.02.
    pub xtsswgs: R,
    /// The power dependence of JTSSWGD on temperature, `xtsswgd`; default `xtsswgs`.
    pub xtsswgd: Option<R>,
    /// The temperature coefficient for NJTS, `tnjts`; default 0.
    pub tnjts: R,
    /// The temperature coefficient for NJTSSW, `tnjtssw`; default 0.
    pub tnjtssw: R,
    /// The temperature coefficient for NJTSSWG, `tnjtsswg`; default 0.
    pub tnjtsswg: R,
    /// The source bottom trap-assisted voltage dependent parameter, `vtss`, in volts;
    /// default 10.
    pub vtss: R,
    /// The drain bottom trap-assisted voltage dependent parameter, `vtsd`, in volts;
    /// default `vtss`.
    pub vtsd: Option<R>,
    /// The source STI sidewall trap-assisted voltage dependent parameter, `vtssws`, in
    /// volts; default 10.
    pub vtssws: R,
    /// The drain STI sidewall trap-assisted voltage dependent parameter, `vtsswd`, in
    /// volts; default `vtssws`.
    pub vtsswd: Option<R>,
    /// The source gate-edge sidewall trap-assisted voltage dependent parameter,
    /// `vtsswgs`, in volts; default 10.
    pub vtsswgs: R,
    /// The drain gate-edge sidewall trap-assisted voltage dependent parameter,
    /// `vtsswgd`, in volts; default `vtsswgs`.
    pub vtsswgd: Option<R>,
    /// The least conductance of each body resistance, `gbmin`, in siemens; default
    /// 1e-12.
    pub gbmin: R,
    /// The resistance from the drain's body node to the bulk, `rbdb`, in ohms; default
    /// 50.
    pub rbdb: R,
    /// The resistance from the channel's body node to the bulk, `rbpb`, in ohms;
    /// default 50.
    pub rbpb: R,
    /// The resistance from the source's body node to the bulk, `rbsb`, in ohms; default
    /// 50.
    pub rbsb: R,
    /// The resistance from the channel's body node to the source's, `rbps`, in ohms;
    /// default 50.
    pub rbps: R,
    /// The resistance from the channel's body node to the drain's, `rbpd`, in ohms;
    /// default 50.
    pub rbpd: R,
    /// The scaling of `rbps` at `rbodymod` 2, `rbps0`, in ohms; default 50.
    pub rbps0: R,
    /// The body resistance RBPS L scaling, `rbpsl`; default 0.
    pub rbpsl: R,
    /// The body resistance RBPS W scaling, `rbpsw`; default 0.
    pub rbpsw: R,
    /// The body resistance RBPS NF scaling, `rbpsnf`; default 0.
    pub rbpsnf: R,
    /// The scaling of `rbpd` at `rbodymod` 2, `rbpd0`, in ohms; default 50.
    pub rbpd0: R,
    /// The body resistance RBPD L scaling, `rbpdl`; default 0.
    pub rbpdl: R,
    /// The body resistance RBPD W scaling, `rbpdw`; default 0.
    pub rbpdw: R,
    /// The body resistance RBPD NF scaling, `rbpdnf`; default 0.
    pub rbpdnf: R,
    /// The scaling of `rbpbx` at `rbodymod` 2, `rbpbx0`, in ohms; default 100.
    pub rbpbx0: R,
    /// The body resistance RBPBX L scaling, `rbpbxl`; default 0.
    pub rbpbxl: R,
    /// The body resistance RBPBX W scaling, `rbpbxw`; default 0.
    pub rbpbxw: R,
    /// The body resistance RBPBX NF scaling, `rbpbxnf`; default 0.
    pub rbpbxnf: R,
    /// The scaling of `rbpby` at `rbodymod` 2, `rbpby0`, in ohms; default 100.
    pub rbpby0: R,
    /// The body resistance RBPBY L scaling, `rbpbyl`; default 0.
    pub rbpbyl: R,
    /// The body resistance RBPBY W scaling, `rbpbyw`; default 0.
    pub rbpbyw: R,
    /// The body resistance RBPBY NF scaling, `rbpbynf`; default 0.
    pub rbpbynf: R,
    /// The scaling of `rbsbx` at `rbodymod` 2, `rbsbx0`, in ohms; default 100.
    pub rbsbx0: R,
    /// The scaling of `rbsby` at `rbodymod` 2, `rbsby0`, in ohms; default 100.
    pub rbsby0: R,
    /// The scaling of `rbdbx` at `rbodymod` 2, `rbdbx0`, in ohms; default 100.
    pub rbdbx0: R,
    /// The scaling of `rbdby` at `rbodymod` 2, `rbdby0`, in ohms; default 100.
    pub rbdby0: R,
    /// The body resistance RBSDBX L scaling, `rbsdbxl`; default 0.
    pub rbsdbxl: R,
    /// The body resistance RBSDBX W scaling, `rbsdbxw`; default 0.
    pub rbsdbxw: R,
    /// The body resistance RBSDBX NF scaling, `rbsdbxnf`; default 0.
    pub rbsdbxnf: R,
    /// The body resistance RBSDBY L scaling, `rbsdbyl`; default 0.
    pub rbsdbyl: R,
    /// The body resistance RBSDBY W scaling, `rbsdbyw`; default 0.
    pub rbsdbyw: R,
    /// The body resistance RBSDBY NF scaling, `rbsdbynf`; default 0.
    pub rbsdbynf: R,
    /// The reference distance between OD edge to poly of one side, `saref`, in meters;
    /// default 1e-06.
    pub saref: R,
    /// The reference distance between OD edge to poly of the other side, `sbref`, in
    /// meters; default 1e-06.
    pub sbref: R,
    /// The width parameter for stress effect, `wlod`, in meters; default 0.
    pub wlod: R,
    /// The mobility degradation/enhancement coefficient for LOD, `ku0`, in meters;
    /// default 0.
    pub ku0: R,
    /// The saturation velocity degradation/enhancement parameter for LOD, `kvsat`;
    /// default 0.
    pub kvsat: R,
    /// The threshold degradation/enhancement parameter for LOD, `kvth0`, in volt
    /// meters; default 0.
    pub kvth0: R,
    /// The temperature coefficient of KU0, `tku0`; default 0.
    pub tku0: R,
    /// The length parameter for u0 LOD effect, `llodku0`; default 0.
    pub llodku0: R,
    /// The width parameter for u0 LOD effect, `wlodku0`; default 0.
    pub wlodku0: R,
    /// The length parameter for vth LOD effect, `llodvth`; default 0.
    pub llodvth: R,
    /// The width parameter for vth LOD effect, `wlodvth`; default 0.
    pub wlodvth: R,
    /// The length dependence of ku0, `lku0`, in the manual's units; default 0.
    pub lku0: R,
    /// The width dependence of ku0, `wku0`, in the manual's units; default 0.
    pub wku0: R,
    /// The cross-term dependence of ku0, `pku0`, in the manual's units; default 0.
    pub pku0: R,
    /// The length dependence of kvth0, `lkvth0`, in the manual's units; default 0.
    pub lkvth0: R,
    /// The width dependence of kvth0, `wkvth0`, in the manual's units; default 0.
    pub wkvth0: R,
    /// The cross-term dependence of kvth0, `pkvth0`, in the manual's units; default 0.
    pub pkvth0: R,
    /// The K2 shift factor related to stress effect on vth, `stk2`, in meters; default
    /// 0.
    pub stk2: R,
    /// The K2 shift modification factor for stress effect, `lodk2`; default 1.
    pub lodk2: R,
    /// The eta0 shift factor related to stress effect on vth, `steta0`, in meters;
    /// default 0.
    pub steta0: R,
    /// The eta0 shift modification factor for stress effect, `lodeta0`; default 1.
    pub lodeta0: R,
    /// The coefficient for SCB, `web`; default 0.
    pub web: R,
    /// The coefficient for SCC, `wec`; default 0.
    pub wec: R,
    /// The threshold shift factor for well proximity effect, `kvth0we`, in volts;
    /// default 0.
    pub kvth0we: R,
    /// The length dependence of `kvth0we`, `lkvth0we`: its coefficient of the
    /// reciprocal of the effective length, in the unit of `kvth0we` times meters;
    /// default 0.
    pub lkvth0we: R,
    /// The width dependence of `kvth0we`, `wkvth0we`: its coefficient of the reciprocal
    /// of the effective width, in the unit of `kvth0we` times meters; default 0.
    pub wkvth0we: R,
    /// The area dependence of `kvth0we`, `pkvth0we`: its coefficient of the reciprocal
    /// of the effective length times the effective width, in the unit of `kvth0we`
    /// times square meters; default 0.
    pub pkvth0we: R,
    /// The K2 shift factor for well proximity effect, `k2we`; default 0.
    pub k2we: R,
    /// The length dependence of `k2we`, `lk2we`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `k2we` times meters; default 0.
    pub lk2we: R,
    /// The width dependence of `k2we`, `wk2we`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `k2we` times meters; default 0.
    pub wk2we: R,
    /// The area dependence of `k2we`, `pk2we`: its coefficient of the reciprocal of the
    /// effective length times the effective width, in the unit of `k2we` times square
    /// meters; default 0.
    pub pk2we: R,
    /// The mobility degradation factor for well proximity effect, `ku0we`; default 0.
    pub ku0we: R,
    /// The length dependence of `ku0we`, `lku0we`: its coefficient of the reciprocal of
    /// the effective length, in the unit of `ku0we` times meters; default 0.
    pub lku0we: R,
    /// The width dependence of `ku0we`, `wku0we`: its coefficient of the reciprocal of
    /// the effective width, in the unit of `ku0we` times meters; default 0.
    pub wku0we: R,
    /// The area dependence of `ku0we`, `pku0we`: its coefficient of the reciprocal of
    /// the effective length times the effective width, in the unit of `ku0we` times
    /// square meters; default 0.
    pub pku0we: R,
    /// The reference distance to calculate SCA, SCB and SCC, `scref`, in meters;
    /// default 1e-06.
    pub scref: R,
    /// The flicker noise parameter, `noia`, in the manual's units; default 6.25e41 for
    /// an n-channel and 6.188e40 for a p-channel.
    pub noia: Option<R>,
    /// The flicker noise parameter, `noib`, in the manual's units; default 3.125e26 for
    /// an n-channel and 1.5e25 for a p-channel.
    pub noib: Option<R>,
    /// The flicker noise parameter, `noic`, in the manual's units; default 8.75e+09.
    pub noic: R,
    /// The thermal noise parameter, `tnoia`; default 1.5.
    pub tnoia: R,
    /// The thermal noise parameter, `tnoib`; default 3.5.
    pub tnoib: R,
    /// The thermal noise coefficient, `rnoia`; default 0.577.
    pub rnoia: R,
    /// The thermal noise coefficient, `rnoib`; default 0.5164.
    pub rnoib: R,
    /// The thermal noise parameter, `ntnoi`; default 1.
    pub ntnoi: R,
    /// The flicker noise parameter, `em`, in volts per meter; default 4.1e+07.
    pub em: R,
    /// The flicker noise frequency exponent, `ef`; default 1.
    pub ef: R,
    /// The flicker noise exponent, `af`; default 1.
    pub af: R,
    /// The flicker noise coefficient, `kf`, in the manual's units; default 0.
    pub kf: R,
}

impl<R: Real> Default for Bsim4Parameters<R> {
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
            fingers: R::constant(1.0),
            drain_area: None,
            source_area: None,
            drain_perimeter: None,
            source_perimeter: None,
            drain_squares: None,
            source_squares: None,
            diffusion_edge_a: R::constant(0.0),
            diffusion_edge_b: R::constant(0.0),
            finger_spacing: None,
            threshold_adjustment: R::constant(0.0),
            mobility_multiplier: R::constant(1.0),
            body_resistance_model: BodyResistance::Zero,
            gate_resistance_model: GateResistance::Zero,
            perimeter_model: Perimeter::One,
            drain_source_resistance_model: DrainSourceResistance::Zero,
            flicker_noise_model: FlickerNoise::One,
            thermal_noise_model: ThermalNoise::Zero,
            check_parameters: true,
            charge_partition: 0.0,
            nominal_temperature: R::constant(REFERENCE_TEMPERATURE),
            toxe: None,
            toxp: None,
            toxm: None,
            toxref: R::constant(3.0e-9),
            dtox: R::constant(0.0),
            epsrox: R::constant(3.9),
            cdsc: R::constant(2.4e-4),
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
            nfactor: R::constant(1.0),
            lnfactor: R::constant(0.0),
            wnfactor: R::constant(0.0),
            pnfactor: R::constant(0.0),
            xj: R::constant(1.5e-7),
            lxj: R::constant(0.0),
            wxj: R::constant(0.0),
            pxj: R::constant(0.0),
            vsat: R::constant(80000.0),
            lvsat: R::constant(0.0),
            wvsat: R::constant(0.0),
            pvsat: R::constant(0.0),
            at: R::constant(33000.0),
            lat: R::constant(0.0),
            wat: R::constant(0.0),
            pat: R::constant(0.0),
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
            keta: R::constant(-0.047),
            lketa: R::constant(0.0),
            wketa: R::constant(0.0),
            pketa: R::constant(0.0),
            nsub: R::constant(6.0e22),
            lnsub: R::constant(0.0),
            wnsub: R::constant(0.0),
            pnsub: R::constant(0.0),
            ndep: None,
            lndep: R::constant(0.0),
            wndep: R::constant(0.0),
            pndep: R::constant(0.0),
            nsd: R::constant(1.0e26),
            lnsd: R::constant(0.0),
            wnsd: R::constant(0.0),
            pnsd: R::constant(0.0),
            phin: R::constant(0.0),
            lphin: R::constant(0.0),
            wphin: R::constant(0.0),
            pphin: R::constant(0.0),
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
            k1: None,
            lk1: R::constant(0.0),
            wk1: R::constant(0.0),
            pk1: R::constant(0.0),
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
            dvtp0: R::constant(0.0),
            ldvtp0: R::constant(0.0),
            wdvtp0: R::constant(0.0),
            pdvtp0: R::constant(0.0),
            dvtp1: R::constant(0.0),
            ldvtp1: R::constant(0.0),
            wdvtp1: R::constant(0.0),
            pdvtp1: R::constant(0.0),
            lpe0: R::constant(1.74e-7),
            llpe0: R::constant(0.0),
            wlpe0: R::constant(0.0),
            plpe0: R::constant(0.0),
            lpeb: R::constant(0.0),
            llpeb: R::constant(0.0),
            wlpeb: R::constant(0.0),
            plpeb: R::constant(0.0),
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
            drout: R::constant(0.56),
            ldrout: R::constant(0.0),
            wdrout: R::constant(0.0),
            pdrout: R::constant(0.0),
            dsub: None,
            ldsub: R::constant(0.0),
            wdsub: R::constant(0.0),
            pdsub: R::constant(0.0),
            vth0: None,
            lvth0: R::constant(0.0),
            wvth0: R::constant(0.0),
            pvth0: R::constant(0.0),
            ua: None,
            lua: R::constant(0.0),
            wua: R::constant(0.0),
            pua: R::constant(0.0),
            ua1: R::constant(1.0e-9),
            lua1: R::constant(0.0),
            wua1: R::constant(0.0),
            pua1: R::constant(0.0),
            ub: R::constant(1.0e-19),
            lub: R::constant(0.0),
            wub: R::constant(0.0),
            r#pub: R::constant(0.0),
            ub1: R::constant(-1.0e-18),
            lub1: R::constant(0.0),
            wub1: R::constant(0.0),
            pub1: R::constant(0.0),
            uc: None,
            luc: R::constant(0.0),
            wuc: R::constant(0.0),
            puc: R::constant(0.0),
            uc1: None,
            luc1: R::constant(0.0),
            wuc1: R::constant(0.0),
            puc1: R::constant(0.0),
            ud: R::constant(1.0e14),
            lud: R::constant(0.0),
            wud: R::constant(0.0),
            pud: R::constant(0.0),
            ud1: R::constant(0.0),
            lud1: R::constant(0.0),
            wud1: R::constant(0.0),
            pud1: R::constant(0.0),
            up: R::constant(0.0),
            lup: R::constant(0.0),
            wup: R::constant(0.0),
            pup: R::constant(0.0),
            lp: R::constant(1.0e-8),
            llp: R::constant(0.0),
            wlp: R::constant(0.0),
            plp: R::constant(0.0),
            u0: None,
            lu0: R::constant(0.0),
            wu0: R::constant(0.0),
            pu0: R::constant(0.0),
            eu: None,
            leu: R::constant(0.0),
            weu: R::constant(0.0),
            peu: R::constant(0.0),
            ute: R::constant(-1.5),
            lute: R::constant(0.0),
            wute: R::constant(0.0),
            pute: R::constant(0.0),
            voff: R::constant(-0.08),
            lvoff: R::constant(0.0),
            wvoff: R::constant(0.0),
            pvoff: R::constant(0.0),
            minv: R::constant(0.0),
            lminv: R::constant(0.0),
            wminv: R::constant(0.0),
            pminv: R::constant(0.0),
            voffl: R::constant(0.0),
            cgso: None,
            cgdo: None,
            cgbo: None,
            delta: R::constant(0.01),
            ldelta: R::constant(0.0),
            wdelta: R::constant(0.0),
            pdelta: R::constant(0.0),
            rsh: R::constant(0.0),
            rdsw: R::constant(200.0),
            lrdsw: R::constant(0.0),
            wrdsw: R::constant(0.0),
            prdsw: R::constant(0.0),
            rdswmin: R::constant(0.0),
            rsw: R::constant(100.0),
            lrsw: R::constant(0.0),
            wrsw: R::constant(0.0),
            prsw: R::constant(0.0),
            rdw: R::constant(100.0),
            lrdw: R::constant(0.0),
            wrdw: R::constant(0.0),
            prdw: R::constant(0.0),
            rdwmin: R::constant(0.0),
            rswmin: R::constant(0.0),
            prwg: R::constant(1.0),
            lprwg: R::constant(0.0),
            wprwg: R::constant(0.0),
            pprwg: R::constant(0.0),
            prwb: R::constant(0.0),
            lprwb: R::constant(0.0),
            wprwb: R::constant(0.0),
            pprwb: R::constant(0.0),
            prt: R::constant(0.0),
            lprt: R::constant(0.0),
            wprt: R::constant(0.0),
            pprt: R::constant(0.0),
            eta0: R::constant(0.08),
            leta0: R::constant(0.0),
            weta0: R::constant(0.0),
            peta0: R::constant(0.0),
            etab: R::constant(-0.07),
            letab: R::constant(0.0),
            wetab: R::constant(0.0),
            petab: R::constant(0.0),
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
            fprout: R::constant(0.0),
            lfprout: R::constant(0.0),
            wfprout: R::constant(0.0),
            pfprout: R::constant(0.0),
            pdits: R::constant(0.0),
            lpdits: R::constant(0.0),
            wpdits: R::constant(0.0),
            ppdits: R::constant(0.0),
            pditsl: R::constant(0.0),
            pditsd: R::constant(0.0),
            lpditsd: R::constant(0.0),
            wpditsd: R::constant(0.0),
            ppditsd: R::constant(0.0),
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
            jss: R::constant(0.0001),
            jsws: R::constant(0.0),
            jswgs: R::constant(0.0),
            pbs: R::constant(1.0),
            njs: R::constant(1.0),
            xtis: R::constant(3.0),
            mjs: R::constant(0.5),
            pbsws: R::constant(1.0),
            mjsws: R::constant(0.33),
            pbswgs: None,
            mjswgs: None,
            cjs: R::constant(0.0005),
            cjsws: R::constant(5.0e-10),
            cjswgs: None,
            jsd: None,
            jswd: None,
            jswgd: None,
            pbd: None,
            njd: None,
            xtid: None,
            mjd: None,
            pbswd: None,
            mjswd: None,
            pbswgd: None,
            mjswgd: None,
            cjd: None,
            cjswd: None,
            cjswgd: None,
            vfbcv: R::constant(-1.0),
            lvfbcv: R::constant(0.0),
            wvfbcv: R::constant(0.0),
            pvfbcv: R::constant(0.0),
            vfb: None,
            lvfb: R::constant(0.0),
            wvfb: R::constant(0.0),
            pvfb: R::constant(0.0),
            tpb: R::constant(0.0),
            tcj: R::constant(0.0),
            tpbsw: R::constant(0.0),
            tcjsw: R::constant(0.0),
            tpbswg: R::constant(0.0),
            tcjswg: R::constant(0.0),
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
            dmcg: R::constant(0.0),
            dmci: None,
            dmdg: R::constant(0.0),
            dmcgt: R::constant(0.0),
            xgw: R::constant(0.0),
            xgl: R::constant(0.0),
            rshg: R::constant(0.1),
            ngcon: R::constant(1.0),
            xrcrg1: R::constant(12.0),
            lxrcrg1: R::constant(0.0),
            wxrcrg1: R::constant(0.0),
            pxrcrg1: R::constant(0.0),
            xrcrg2: R::constant(1.0),
            lxrcrg2: R::constant(0.0),
            wxrcrg2: R::constant(0.0),
            pxrcrg2: R::constant(0.0),
            lambda: None,
            llambda: R::constant(0.0),
            wlambda: R::constant(0.0),
            plambda: R::constant(0.0),
            vtl: None,
            lvtl: R::constant(0.0),
            wvtl: R::constant(0.0),
            pvtl: R::constant(0.0),
            lc: R::constant(5.0e-9),
            xn: R::constant(3.0),
            lxn: R::constant(0.0),
            wxn: R::constant(0.0),
            pxn: R::constant(0.0),
            vfbsdoff: R::constant(0.0),
            lvfbsdoff: R::constant(0.0),
            wvfbsdoff: R::constant(0.0),
            pvfbsdoff: R::constant(0.0),
            tvfbsdoff: R::constant(0.0),
            ltvfbsdoff: R::constant(0.0),
            wtvfbsdoff: R::constant(0.0),
            ptvfbsdoff: R::constant(0.0),
            tvoff: R::constant(0.0),
            ltvoff: R::constant(0.0),
            wtvoff: R::constant(0.0),
            ptvoff: R::constant(0.0),
            lintnoi: R::constant(0.0),
            lint: R::constant(0.0),
            ll: R::constant(0.0),
            llc: None,
            lln: R::constant(1.0),
            lw: R::constant(0.0),
            lwc: None,
            lwn: R::constant(1.0),
            lwl: R::constant(0.0),
            lwlc: None,
            wr: R::constant(1.0),
            lwr: R::constant(0.0),
            wwr: R::constant(0.0),
            pwr: R::constant(0.0),
            wint: R::constant(0.0),
            dwg: R::constant(0.0),
            ldwg: R::constant(0.0),
            wdwg: R::constant(0.0),
            pdwg: R::constant(0.0),
            dwb: R::constant(0.0),
            ldwb: R::constant(0.0),
            wdwb: R::constant(0.0),
            pdwb: R::constant(0.0),
            wl: R::constant(0.0),
            wlc: None,
            wln: R::constant(1.0),
            ww: R::constant(0.0),
            wwc: None,
            wwn: R::constant(1.0),
            wwl: R::constant(0.0),
            wwlc: None,
            b0: R::constant(0.0),
            lb0: R::constant(0.0),
            wb0: R::constant(0.0),
            pb0: R::constant(0.0),
            b1: R::constant(0.0),
            lb1: R::constant(0.0),
            wb1: R::constant(0.0),
            pb1: R::constant(0.0),
            cgsl: R::constant(0.0),
            lcgsl: R::constant(0.0),
            wcgsl: R::constant(0.0),
            pcgsl: R::constant(0.0),
            cgdl: R::constant(0.0),
            lcgdl: R::constant(0.0),
            wcgdl: R::constant(0.0),
            pcgdl: R::constant(0.0),
            ckappas: R::constant(0.6),
            lckappas: R::constant(0.0),
            wckappas: R::constant(0.0),
            pckappas: R::constant(0.0),
            ckappad: None,
            lckappad: R::constant(0.0),
            wckappad: R::constant(0.0),
            pckappad: R::constant(0.0),
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
            dwc: None,
            dlc: None,
            xw: R::constant(0.0),
            xl: R::constant(0.0),
            dlcig: None,
            dwj: None,
            alpha0: R::constant(0.0),
            lalpha0: R::constant(0.0),
            walpha0: R::constant(0.0),
            palpha0: R::constant(0.0),
            alpha1: R::constant(0.0),
            lalpha1: R::constant(0.0),
            walpha1: R::constant(0.0),
            palpha1: R::constant(0.0),
            beta0: R::constant(0.0),
            lbeta0: R::constant(0.0),
            wbeta0: R::constant(0.0),
            pbeta0: R::constant(0.0),
            agidl: R::constant(0.0),
            lagidl: R::constant(0.0),
            wagidl: R::constant(0.0),
            pagidl: R::constant(0.0),
            bgidl: R::constant(2.3e9),
            lbgidl: R::constant(0.0),
            wbgidl: R::constant(0.0),
            pbgidl: R::constant(0.0),
            cgidl: R::constant(0.5),
            lcgidl: R::constant(0.0),
            wcgidl: R::constant(0.0),
            pcgidl: R::constant(0.0),
            egidl: R::constant(0.8),
            legidl: R::constant(0.0),
            wegidl: R::constant(0.0),
            pegidl: R::constant(0.0),
            aigc: None,
            laigc: R::constant(0.0),
            waigc: R::constant(0.0),
            paigc: R::constant(0.0),
            bigc: None,
            lbigc: R::constant(0.0),
            wbigc: R::constant(0.0),
            pbigc: R::constant(0.0),
            cigc: None,
            lcigc: R::constant(0.0),
            wcigc: R::constant(0.0),
            pcigc: R::constant(0.0),
            aigsd: None,
            laigsd: R::constant(0.0),
            waigsd: R::constant(0.0),
            paigsd: R::constant(0.0),
            bigsd: None,
            lbigsd: R::constant(0.0),
            wbigsd: R::constant(0.0),
            pbigsd: R::constant(0.0),
            cigsd: None,
            lcigsd: R::constant(0.0),
            wcigsd: R::constant(0.0),
            pcigsd: R::constant(0.0),
            aigbacc: R::constant(0.0136),
            laigbacc: R::constant(0.0),
            waigbacc: R::constant(0.0),
            paigbacc: R::constant(0.0),
            bigbacc: R::constant(1.71e-3),
            lbigbacc: R::constant(0.0),
            wbigbacc: R::constant(0.0),
            pbigbacc: R::constant(0.0),
            cigbacc: R::constant(0.075),
            lcigbacc: R::constant(0.0),
            wcigbacc: R::constant(0.0),
            pcigbacc: R::constant(0.0),
            aigbinv: R::constant(0.0111),
            laigbinv: R::constant(0.0),
            waigbinv: R::constant(0.0),
            paigbinv: R::constant(0.0),
            bigbinv: R::constant(9.49e-4),
            lbigbinv: R::constant(0.0),
            wbigbinv: R::constant(0.0),
            pbigbinv: R::constant(0.0),
            cigbinv: R::constant(0.006),
            lcigbinv: R::constant(0.0),
            wcigbinv: R::constant(0.0),
            pcigbinv: R::constant(0.0),
            nigc: R::constant(1.0),
            lnigc: R::constant(0.0),
            wnigc: R::constant(0.0),
            pnigc: R::constant(0.0),
            nigbinv: R::constant(3.0),
            lnigbinv: R::constant(0.0),
            wnigbinv: R::constant(0.0),
            pnigbinv: R::constant(0.0),
            nigbacc: R::constant(1.0),
            lnigbacc: R::constant(0.0),
            wnigbacc: R::constant(0.0),
            pnigbacc: R::constant(0.0),
            ntox: R::constant(1.0),
            lntox: R::constant(0.0),
            wntox: R::constant(0.0),
            pntox: R::constant(0.0),
            eigbinv: R::constant(1.1),
            leigbinv: R::constant(0.0),
            weigbinv: R::constant(0.0),
            peigbinv: R::constant(0.0),
            pigcd: R::constant(1.0),
            lpigcd: R::constant(0.0),
            wpigcd: R::constant(0.0),
            ppigcd: R::constant(0.0),
            poxedge: R::constant(1.0),
            lpoxedge: R::constant(0.0),
            wpoxedge: R::constant(0.0),
            ppoxedge: R::constant(0.0),
            ijthdfwd: None,
            ijthsfwd: R::constant(0.1),
            ijthdrev: None,
            ijthsrev: R::constant(0.1),
            xjbvd: None,
            xjbvs: R::constant(1.0),
            bvd: None,
            bvs: R::constant(10.0),
            jtss: R::constant(0.0),
            jtsd: None,
            jtssws: R::constant(0.0),
            jtsswd: None,
            jtsswgs: R::constant(0.0),
            jtsswgd: None,
            njts: R::constant(20.0),
            njtssw: R::constant(20.0),
            njtsswg: R::constant(20.0),
            xtss: R::constant(0.02),
            xtsd: None,
            xtssws: R::constant(0.02),
            xtsswd: None,
            xtsswgs: R::constant(0.02),
            xtsswgd: None,
            tnjts: R::constant(0.0),
            tnjtssw: R::constant(0.0),
            tnjtsswg: R::constant(0.0),
            vtss: R::constant(10.0),
            vtsd: None,
            vtssws: R::constant(10.0),
            vtsswd: None,
            vtsswgs: R::constant(10.0),
            vtsswgd: None,
            gbmin: R::constant(1.0e-12),
            rbdb: R::constant(50.0),
            rbpb: R::constant(50.0),
            rbsb: R::constant(50.0),
            rbps: R::constant(50.0),
            rbpd: R::constant(50.0),
            rbps0: R::constant(50.0),
            rbpsl: R::constant(0.0),
            rbpsw: R::constant(0.0),
            rbpsnf: R::constant(0.0),
            rbpd0: R::constant(50.0),
            rbpdl: R::constant(0.0),
            rbpdw: R::constant(0.0),
            rbpdnf: R::constant(0.0),
            rbpbx0: R::constant(100.0),
            rbpbxl: R::constant(0.0),
            rbpbxw: R::constant(0.0),
            rbpbxnf: R::constant(0.0),
            rbpby0: R::constant(100.0),
            rbpbyl: R::constant(0.0),
            rbpbyw: R::constant(0.0),
            rbpbynf: R::constant(0.0),
            rbsbx0: R::constant(100.0),
            rbsby0: R::constant(100.0),
            rbdbx0: R::constant(100.0),
            rbdby0: R::constant(100.0),
            rbsdbxl: R::constant(0.0),
            rbsdbxw: R::constant(0.0),
            rbsdbxnf: R::constant(0.0),
            rbsdbyl: R::constant(0.0),
            rbsdbyw: R::constant(0.0),
            rbsdbynf: R::constant(0.0),
            saref: R::constant(1.0e-6),
            sbref: R::constant(1.0e-6),
            wlod: R::constant(0.0),
            ku0: R::constant(0.0),
            kvsat: R::constant(0.0),
            kvth0: R::constant(0.0),
            tku0: R::constant(0.0),
            llodku0: R::constant(0.0),
            wlodku0: R::constant(0.0),
            llodvth: R::constant(0.0),
            wlodvth: R::constant(0.0),
            lku0: R::constant(0.0),
            wku0: R::constant(0.0),
            pku0: R::constant(0.0),
            lkvth0: R::constant(0.0),
            wkvth0: R::constant(0.0),
            pkvth0: R::constant(0.0),
            stk2: R::constant(0.0),
            lodk2: R::constant(1.0),
            steta0: R::constant(0.0),
            lodeta0: R::constant(1.0),
            web: R::constant(0.0),
            wec: R::constant(0.0),
            kvth0we: R::constant(0.0),
            lkvth0we: R::constant(0.0),
            wkvth0we: R::constant(0.0),
            pkvth0we: R::constant(0.0),
            k2we: R::constant(0.0),
            lk2we: R::constant(0.0),
            wk2we: R::constant(0.0),
            pk2we: R::constant(0.0),
            ku0we: R::constant(0.0),
            lku0we: R::constant(0.0),
            wku0we: R::constant(0.0),
            pku0we: R::constant(0.0),
            scref: R::constant(1.0e-6),
            noia: None,
            noib: None,
            noic: R::constant(8.75e9),
            tnoia: R::constant(1.5),
            tnoib: R::constant(3.5),
            rnoia: R::constant(0.577),
            rnoib: R::constant(0.5164),
            ntnoi: R::constant(1.0),
            em: R::constant(4.1e7),
            ef: R::constant(1.0),
            af: R::constant(1.0),
            kf: R::constant(0.0),
        }
    }
}

/// The binned parameters the modes jolty models read, at one size: each the card's value
/// plus its length, width and area dependences at the reciprocals of the effective length,
/// width and their product.
#[derive(Debug, Clone, Copy)]
pub(super) struct Binned<R: Real> {
    pub(super) cdsc: R,
    pub(super) cdscb: R,
    pub(super) cdscd: R,
    pub(super) cit: R,
    pub(super) nfactor: R,
    pub(super) xj: R,
    pub(super) vsat: R,
    pub(super) at: R,
    pub(super) a0: R,
    pub(super) ags: R,
    pub(super) a1: R,
    pub(super) a2: R,
    pub(super) keta: R,
    pub(super) nsub: R,
    pub(super) ndep: R,
    pub(super) nsd: R,
    pub(super) phin: R,
    pub(super) ngate: R,
    pub(super) gamma1: Option<R>,
    pub(super) gamma2: Option<R>,
    pub(super) vbx: Option<R>,
    pub(super) vbm: R,
    pub(super) xt: R,
    pub(super) vfb: Option<R>,
    pub(super) k1: Option<R>,
    pub(super) kt1: R,
    pub(super) kt1l: R,
    pub(super) k2: Option<R>,
    pub(super) kt2: R,
    pub(super) k3: R,
    pub(super) k3b: R,
    pub(super) w0: R,
    pub(super) lpe0: R,
    pub(super) lpeb: R,
    pub(super) dvtp0: R,
    pub(super) dvtp1: R,
    pub(super) dvt0: R,
    pub(super) dvt1: R,
    pub(super) dvt2: R,
    pub(super) dvt0w: R,
    pub(super) dvt1w: R,
    pub(super) dvt2w: R,
    pub(super) drout: R,
    pub(super) dsub: R,
    pub(super) vth0: Option<R>,
    pub(super) ua: R,
    pub(super) ua1: R,
    pub(super) ub: R,
    pub(super) ub1: R,
    pub(super) uc: R,
    pub(super) uc1: R,
    pub(super) ud: R,
    pub(super) ud1: R,
    pub(super) up: R,
    pub(super) lp: R,
    pub(super) u0: R,
    pub(super) ute: R,
    pub(super) voff: R,
    pub(super) tvoff: R,
    pub(super) minv: R,
    pub(super) fprout: R,
    pub(super) pdits: R,
    pub(super) pditsd: R,
    pub(super) delta: R,
    pub(super) rdsw: R,
    pub(super) rdw: R,
    pub(super) rsw: R,
    pub(super) prwg: R,
    pub(super) prwb: R,
    pub(super) prt: R,
    pub(super) eta0: R,
    pub(super) etab: R,
    pub(super) pclm: R,
    pub(super) pdiblc1: R,
    pub(super) pdiblc2: R,
    pub(super) pdiblcb: R,
    pub(super) pscbe1: R,
    pub(super) pscbe2: R,
    pub(super) pvag: R,
    pub(super) wr: R,
    pub(super) dwg: R,
    pub(super) dwb: R,
    pub(super) b0: R,
    pub(super) b1: R,
    pub(super) alpha0: R,
    pub(super) alpha1: R,
    pub(super) beta0: R,
    pub(super) agidl: R,
    pub(super) bgidl: R,
    pub(super) cgidl: R,
    pub(super) egidl: R,
    pub(super) lambda: R,
    pub(super) vtl: R,
    pub(super) xn: R,
    pub(super) cgsl: R,
    pub(super) cgdl: R,
    pub(super) ckappas: R,
    pub(super) ckappad: R,
    pub(super) cf: R,
    pub(super) clc: R,
    pub(super) cle: R,
    pub(super) acde: R,
    pub(super) xrcrg1: R,
    pub(super) xrcrg2: R,
    pub(super) moin: R,
    pub(super) noff: R,
    pub(super) voffcv: R,
}

impl<R: Real> Bsim4Parameters<R> {
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
            nfactor: self.nfactor
                + self.lnfactor * inverse_length
                + self.wnfactor * inverse_width
                + self.pnfactor * inverse_area,
            xj: self.xj
                + self.lxj * inverse_length
                + self.wxj * inverse_width
                + self.pxj * inverse_area,
            vsat: self.vsat
                + self.lvsat * inverse_length
                + self.wvsat * inverse_width
                + self.pvsat * inverse_area,
            at: self.at
                + self.lat * inverse_length
                + self.wat * inverse_width
                + self.pat * inverse_area,
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
            keta: self.keta
                + self.lketa * inverse_length
                + self.wketa * inverse_width
                + self.pketa * inverse_area,
            nsub: self.nsub
                + self.lnsub * inverse_length
                + self.wnsub * inverse_width
                + self.pnsub * inverse_area,
            ndep: self.ndep.unwrap_or(R::constant(1.7e23))
                + self.lndep * inverse_length
                + self.wndep * inverse_width
                + self.pndep * inverse_area,
            nsd: self.nsd
                + self.lnsd * inverse_length
                + self.wnsd * inverse_width
                + self.pnsd * inverse_area,
            phin: self.phin
                + self.lphin * inverse_length
                + self.wphin * inverse_width
                + self.pphin * inverse_area,
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
            kt1: self.kt1
                + self.lkt1 * inverse_length
                + self.wkt1 * inverse_width
                + self.pkt1 * inverse_area,
            kt1l: self.kt1l
                + self.lkt1l * inverse_length
                + self.wkt1l * inverse_width
                + self.pkt1l * inverse_area,
            k2: self.k2.map(|value| {
                value
                    + self.lk2 * inverse_length
                    + self.wk2 * inverse_width
                    + self.pk2 * inverse_area
            }),
            kt2: self.kt2
                + self.lkt2 * inverse_length
                + self.wkt2 * inverse_width
                + self.pkt2 * inverse_area,
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
            lpe0: self.lpe0
                + self.llpe0 * inverse_length
                + self.wlpe0 * inverse_width
                + self.plpe0 * inverse_area,
            lpeb: self.lpeb
                + self.llpeb * inverse_length
                + self.wlpeb * inverse_width
                + self.plpeb * inverse_area,
            dvtp0: self.dvtp0
                + self.ldvtp0 * inverse_length
                + self.wdvtp0 * inverse_width
                + self.pdvtp0 * inverse_area,
            dvtp1: self.dvtp1
                + self.ldvtp1 * inverse_length
                + self.wdvtp1 * inverse_width
                + self.pdvtp1 * inverse_area,
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
            drout: self.drout
                + self.ldrout * inverse_length
                + self.wdrout * inverse_width
                + self.pdrout * inverse_area,
            dsub: self.dsub.unwrap_or(self.drout)
                + self.ldsub * inverse_length
                + self.wdsub * inverse_width
                + self.pdsub * inverse_area,
            vth0: self.vth0.map(|value| {
                value
                    + self.lvth0 * inverse_length
                    + self.wvth0 * inverse_width
                    + self.pvth0 * inverse_area
            }),
            ua: self.ua.unwrap_or(R::constant(1.0e-9))
                + self.lua * inverse_length
                + self.wua * inverse_width
                + self.pua * inverse_area,
            ua1: self.ua1
                + self.lua1 * inverse_length
                + self.wua1 * inverse_width
                + self.pua1 * inverse_area,
            ub: self.ub
                + self.lub * inverse_length
                + self.wub * inverse_width
                + self.r#pub * inverse_area,
            ub1: self.ub1
                + self.lub1 * inverse_length
                + self.wub1 * inverse_width
                + self.pub1 * inverse_area,
            uc: self.uc.unwrap_or(R::constant(-0.0465e-9))
                + self.luc * inverse_length
                + self.wuc * inverse_width
                + self.puc * inverse_area,
            uc1: self.uc1.unwrap_or(R::constant(-0.056e-9))
                + self.luc1 * inverse_length
                + self.wuc1 * inverse_width
                + self.puc1 * inverse_area,
            ud: self.ud
                + self.lud * inverse_length
                + self.wud * inverse_width
                + self.pud * inverse_area,
            ud1: self.ud1
                + self.lud1 * inverse_length
                + self.wud1 * inverse_width
                + self.pud1 * inverse_area,
            up: self.up
                + self.lup * inverse_length
                + self.wup * inverse_width
                + self.pup * inverse_area,
            lp: self.lp
                + self.llp * inverse_length
                + self.wlp * inverse_width
                + self.plp * inverse_area,
            u0: self
                .u0
                .unwrap_or(R::constant(by_channel(self.channel, 0.067, 0.025)))
                + self.lu0 * inverse_length
                + self.wu0 * inverse_width
                + self.pu0 * inverse_area,
            ute: self.ute
                + self.lute * inverse_length
                + self.wute * inverse_width
                + self.pute * inverse_area,
            voff: self.voff
                + self.lvoff * inverse_length
                + self.wvoff * inverse_width
                + self.pvoff * inverse_area,
            tvoff: self.tvoff
                + self.ltvoff * inverse_length
                + self.wtvoff * inverse_width
                + self.ptvoff * inverse_area,
            minv: self.minv
                + self.lminv * inverse_length
                + self.wminv * inverse_width
                + self.pminv * inverse_area,
            fprout: self.fprout
                + self.lfprout * inverse_length
                + self.wfprout * inverse_width
                + self.pfprout * inverse_area,
            pdits: self.pdits
                + self.lpdits * inverse_length
                + self.wpdits * inverse_width
                + self.ppdits * inverse_area,
            pditsd: self.pditsd
                + self.lpditsd * inverse_length
                + self.wpditsd * inverse_width
                + self.ppditsd * inverse_area,
            delta: self.delta
                + self.ldelta * inverse_length
                + self.wdelta * inverse_width
                + self.pdelta * inverse_area,
            rdsw: self.rdsw
                + self.lrdsw * inverse_length
                + self.wrdsw * inverse_width
                + self.prdsw * inverse_area,
            rdw: self.rdw
                + self.lrdw * inverse_length
                + self.wrdw * inverse_width
                + self.prdw * inverse_area,
            rsw: self.rsw
                + self.lrsw * inverse_length
                + self.wrsw * inverse_width
                + self.prsw * inverse_area,
            prwg: self.prwg
                + self.lprwg * inverse_length
                + self.wprwg * inverse_width
                + self.pprwg * inverse_area,
            prwb: self.prwb
                + self.lprwb * inverse_length
                + self.wprwb * inverse_width
                + self.pprwb * inverse_area,
            prt: self.prt
                + self.lprt * inverse_length
                + self.wprt * inverse_width
                + self.pprt * inverse_area,
            eta0: self.eta0
                + self.leta0 * inverse_length
                + self.weta0 * inverse_width
                + self.peta0 * inverse_area,
            etab: self.etab
                + self.letab * inverse_length
                + self.wetab * inverse_width
                + self.petab * inverse_area,
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
            b0: self.b0
                + self.lb0 * inverse_length
                + self.wb0 * inverse_width
                + self.pb0 * inverse_area,
            b1: self.b1
                + self.lb1 * inverse_length
                + self.wb1 * inverse_width
                + self.pb1 * inverse_area,
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
            agidl: self.agidl
                + self.lagidl * inverse_length
                + self.wagidl * inverse_width
                + self.pagidl * inverse_area,
            bgidl: self.bgidl
                + self.lbgidl * inverse_length
                + self.wbgidl * inverse_width
                + self.pbgidl * inverse_area,
            cgidl: self.cgidl
                + self.lcgidl * inverse_length
                + self.wcgidl * inverse_width
                + self.pcgidl * inverse_area,
            egidl: self.egidl
                + self.legidl * inverse_length
                + self.wegidl * inverse_width
                + self.pegidl * inverse_area,
            lambda: self.lambda.unwrap_or(R::constant(0.0))
                + self.llambda * inverse_length
                + self.wlambda * inverse_width
                + self.plambda * inverse_area,
            vtl: self.vtl.unwrap_or(R::constant(2.0e5))
                + self.lvtl * inverse_length
                + self.wvtl * inverse_width
                + self.pvtl * inverse_area,
            xn: self.xn
                + self.lxn * inverse_length
                + self.wxn * inverse_width
                + self.pxn * inverse_area,
            cgsl: self.cgsl
                + self.lcgsl * inverse_length
                + self.wcgsl * inverse_width
                + self.pcgsl * inverse_area,
            cgdl: self.cgdl
                + self.lcgdl * inverse_length
                + self.wcgdl * inverse_width
                + self.pcgdl * inverse_area,
            ckappas: self.ckappas
                + self.lckappas * inverse_length
                + self.wckappas * inverse_width
                + self.pckappas * inverse_area,
            ckappad: self.ckappad.unwrap_or(self.ckappas)
                + self.lckappad * inverse_length
                + self.wckappad * inverse_width
                + self.pckappad * inverse_area,
            cf: self.cf.unwrap_or(fringe(self.epsrox, self.set_toxe()))
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
            acde: self.acde
                + self.lacde * inverse_length
                + self.wacde * inverse_width
                + self.pacde * inverse_area,
            xrcrg1: self.xrcrg1
                + self.lxrcrg1 * inverse_length
                + self.wxrcrg1 * inverse_width
                + self.pxrcrg1 * inverse_area,
            xrcrg2: self.xrcrg2
                + self.lxrcrg2 * inverse_length
                + self.wxrcrg2 * inverse_width
                + self.pxrcrg2 * inverse_area,
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
