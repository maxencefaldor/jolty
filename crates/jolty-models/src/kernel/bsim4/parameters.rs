//! The parameters of a BSIM4 transistor, by the names of the BSIM4.5 manual, in SI units,
//! with ngspice's defaults.
//!
//! Defaults match ngspice 47 `b4v5set.c`. Parameters whose defaults derive from other
//! fields (such as `vth0` from `vfb`) are optional, and each field description states
//! its derivation rule.
//!
//! Part of jolty's BSIM4, derived from BSIM4.5.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2005 Regents of the University of California. All rights reserved. Used
//! under the BSIM4 terms of use, which the repository's `NOTICE` reproduces.

use jolty_derive::binned_parameters;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{
    BodyResistance, DrainSourceResistance, FlickerNoise, GateResistance, Perimeter, ThermalNoise,
    fringe,
};
use crate::dimension::{
    CAPACITANCE, CONDUCTANCE, CURRENT, LENGTH, MOBILITY, PER_KELVIN, PER_VOLT, RESISTANCE,
    TEMPERATURE, TIME, VOLTAGE, VOLTS_PER_KELVIN,
};
use crate::kernel::bsim3::by_channel;
use crate::kernel::jfet::Channel;
use crate::{Lift, REFERENCE_TEMPERATURE, Real};

binned_parameters! {
    /// Parameters of a [`Bsim4`](super::Bsim4) transistor: instance geometry followed by
    /// the model card. Model-card fields preserve BSIM4.5 manual names, and every binned
    /// field accepts `l`, `w`, and `p` prefixes for length, width, and area terms in SI
    /// units (reciprocal meters rather than micrometers).
    #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
    #[serde(default, deny_unknown_fields)]
    #[schemars(bound = "R: Serialize + JsonSchema")]
    pub struct Bsim4Parameters<R: Real = f64> {
        /// The channel; default `n`.
        pub channel: Channel = Channel::N,
        /// The channel's drawn width, `w`, in meters, of all its fingers together; default
        /// 5 µm.
        pub width: R = R::constant(5e-6) [LENGTH],
        /// The channel's drawn length, `l`, in meters; default 5 µm.
        pub length: R = R::constant(5e-6) [LENGTH],
        /// The number of identical transistors in parallel, `m`; default 1.
        pub multiplier: R = R::constant(1.0),
        /// The number of fingers the channel's width is split into, `nf`; default 1.
        pub fingers: R = R::constant(1.0),
        /// The drain diffusion's area, `ad`, in square meters; default derived from the
        /// fingers and `dmcg`, `dmci` and `dmcgt`.
        pub drain_area: Option<R> = None [LENGTH.pow(2)],
        /// The source diffusion's area, `as`, in square meters; default derived from the
        /// fingers and `dmcg`, `dmci` and `dmcgt`.
        pub source_area: Option<R> = None [LENGTH.pow(2)],
        /// The drain diffusion's perimeter, `pd`, in meters; default derived from the fingers
        /// and `dmcg`, `dmci` and `dmcgt`.
        pub drain_perimeter: Option<R> = None [LENGTH],
        /// The source diffusion's perimeter, `ps`, in meters; default derived from the fingers
        /// and `dmcg`, `dmci` and `dmcgt`.
        pub source_perimeter: Option<R> = None [LENGTH],
        /// The drain diffusion's length in squares, `nrd`, behind which `rsh` sets a
        /// resistance where it is positive; default none, and no resistance.
        pub drain_squares: Option<R> = None,
        /// The source diffusion's length in squares, `nrs`, behind which `rsh` sets a
        /// resistance where it is positive; default none, and no resistance.
        pub source_squares: Option<R> = None,
        /// The distance from the gate to the edge of the diffusion on one side, `sa`, in
        /// meters, which sets the stress of the length of diffusion where it and `sb` are
        /// positive; default 0.
        pub diffusion_edge_a: R = R::constant(0.0) [LENGTH],
        /// The distance from the gate to the edge of the diffusion on the other side, `sb`,
        /// in meters; default 0.
        pub diffusion_edge_b: R = R::constant(0.0) [LENGTH],
        /// The distance between neighboring fingers, `sd`, in meters; default twice `dmcg`.
        pub finger_spacing: Option<R> = None [LENGTH],
        /// A shift of the zero-bias threshold, `delvto`, in volts; default 0.
        pub threshold_adjustment: R = R::constant(0.0) [VOLTAGE],
        /// A factor of the low-field mobility, `mulu0`; default 1.
        pub mobility_multiplier: R = R::constant(1.0),
        /// The body's resistance network, `rbodymod`, 0 or 1; default 0.
        pub body_resistance_model: BodyResistance = BodyResistance::Zero,
        /// The gate's resistance network, `rgatemod`, 0 to 3; default 0.
        pub gate_resistance_model: GateResistance = GateResistance::Zero,
        /// Diffusion perimeter convention (`permod`, 0 or 1) determining whether `ps` and
        /// `pd` include the gate edge; default 1.
        pub perimeter_model: Perimeter = Perimeter::One,
        /// The source and drain resistances, `rdsmod`, 0 or 1; default 0.
        pub drain_source_resistance_model: DrainSourceResistance = DrainSourceResistance::Zero,
        /// The flicker noise model, `fnoimod`, 0 or 1; default 1.
        pub flicker_noise_model: FlickerNoise = FlickerNoise::One,
        /// The channel's thermal noise model, `tnoimod`, 0 or 1; default 0.
        pub thermal_noise_model: ThermalNoise = ThermalNoise::Zero,
        /// Flag enabling range checking and clamping of size-dependent parameters
        /// (`paramchk = 1`); default true.
        pub check_parameters: bool = true,
        /// The share of the channel's charge the drain takes in saturation, `xpart`: 0.4 below
        /// 0.5, 0.5 at 0.5 and none above it; default 0.
        pub charge_partition: f64 = 0.0,
        /// The temperature the parameters were measured at, `tnom`, in kelvins; default 27 °C.
        pub nominal_temperature: R = R::constant(REFERENCE_TEMPERATURE) [TEMPERATURE],
        /// The electrical thickness of the gate oxide, `toxe`, in meters; default 3 nm, or
        /// `toxp` plus `dtox` where `toxp` alone is given.
        pub toxe: Option<R> = None [LENGTH],
        /// The physical thickness of the gate oxide, `toxp`, in meters; default `toxe` less
        /// `dtox` where `toxe` is given, else `toxe`.
        pub toxp: Option<R> = None [LENGTH],
        /// The oxide thickness the parameters were extracted at, `toxm`, in meters; default
        /// `toxe` where it is given, else 3 nm.
        pub toxm: Option<R> = None [LENGTH],
        /// The oxide thickness of reference of the gate tunneling currents, `toxref`, in
        /// meters; default 3e-09.
        pub toxref: R = R::constant(3.0e-9) [LENGTH],
        /// The electrical less the physical thickness of the gate oxide, `dtox`, in meters;
        /// default 0.
        pub dtox: R = R::constant(0.0) [LENGTH],
        /// The dielectric constant of the gate oxide relative to vacuum, `epsrox`; default
        /// 3.9.
        pub epsrox: R = R::constant(3.9),
        /// The coupling capacitance of the drain and the source to the channel, `cdsc`, in
        /// farads per square meter; default 0.00024.
        binned pub cdsc: R = R::constant(2.4e-4) [CAPACITANCE.div(LENGTH.pow(2))],
        /// The body-bias dependence of `cdsc`, `cdscb`, in farads per volt square meter;
        /// default 0.
        binned pub cdscb: R = R::constant(0.0) [CAPACITANCE.div(VOLTAGE.mul(LENGTH.pow(2)))],
        /// The drain-bias dependence of `cdsc`, `cdscd`, in farads per volt square meter;
        /// default 0.
        binned pub cdscd: R = R::constant(0.0) [CAPACITANCE.div(VOLTAGE.mul(LENGTH.pow(2)))],
        /// The interface state capacitance, `cit`, in farads per square meter; default 0.
        binned pub cit: R = R::constant(0.0) [CAPACITANCE.div(LENGTH.pow(2))],
        /// The subthreshold swing coefficient, `nfactor`; default 1.
        binned pub nfactor: R = R::constant(1.0),
        /// The source and drain junctions' depth, `xj`, in meters; default 1.5e-07.
        binned pub xj: R = R::constant(1.5e-7) [LENGTH],
        /// The saturation velocity at tnom, `vsat`, in meters per second; default 80000.
        binned pub vsat: R = R::constant(80000.0) [LENGTH.div(TIME)],
        /// The temperature coefficient of vsat, `at`, in meters per second; default 33000.
        binned pub at: R = R::constant(33000.0) [LENGTH.div(TIME)],
        /// The non-uniform depletion width effect coefficient, `a0`; default 1.
        binned pub a0: R = R::constant(1.0),
        /// The gate bias coefficient of Abulk, `ags`, in per volt; default 0.
        binned pub ags: R = R::constant(0.0) [PER_VOLT],
        /// The non-saturation effect coefficient, `a1`, in per volt; default 0.
        binned pub a1: R = R::constant(0.0) [PER_VOLT],
        /// The non-saturation effect coefficient, `a2`; default 1.
        binned pub a2: R = R::constant(1.0),
        /// The body-bias coefficient of the non-uniform depletion width effect, `keta`, in
        /// per volt; default -0.047.
        binned pub keta: R = R::constant(-0.047) [PER_VOLT],
        /// The substrate doping concentration, `nsub`, in per cubic meter; default 6e+22.
        binned pub nsub: R = R::constant(6.0e22) [LENGTH.pow(-3)],
        /// The channel doping concentration at the depletion edge, `ndep`, in per cubic
        /// meter; default 1.7e23 per cubic meter, or derived from `gamma1` where it alone
        /// is given.
        binned pub ndep: Option<R> = None => R::constant(1.7e23) [LENGTH.pow(-3)],
        /// The source and drain doping concentration, `nsd`, in per cubic meter; default
        /// 1e+26.
        binned pub nsd: R = R::constant(1.0e26) [LENGTH.pow(-3)],
        /// The adjusting parameter for surface potential due to non-uniform vertical
        /// doping, `phin`, in volts; default 0.
        binned pub phin: R = R::constant(0.0) [VOLTAGE],
        /// The poly-gate doping concentration, `ngate`, in per cubic meter; default 0.
        binned pub ngate: R = R::constant(0.0) [LENGTH.pow(-3)],
        /// The body effect near the surface, `gamma1`, in square-root volts; default
        /// derived from `ndep`.
        binned pub gamma1: Option<R> = None [VOLTAGE.sqrt()],
        /// The body effect in the bulk, `gamma2`, in square-root volts; default derived
        /// from `nsub`.
        binned pub gamma2: Option<R> = None [VOLTAGE.sqrt()],
        /// The bulk-source voltage at which the depletion reaches `xt`, `vbx`, in volts;
        /// default derived from `ndep` and `xt`.
        binned pub vbx: Option<R> = None [VOLTAGE],
        /// The most negative bulk-source voltage the threshold's body effect reads, `vbm`,
        /// in volts; default -3.
        binned pub vbm: R = R::constant(-3.0) [VOLTAGE],
        /// The depth of the doping near the surface, `xt`, in meters; default 1.55e-07.
        binned pub xt: R = R::constant(1.55e-7) [LENGTH],
        /// The first-order body effect, `k1`, in square-root volts; default derived from
        /// `gamma1` and `gamma2`, or 0.53 √V where only `k2` is given.
        binned pub k1: Option<R> = None [VOLTAGE.sqrt()],
        /// The temperature coefficient of the threshold, `kt1`, in volts; default -0.11.
        binned pub kt1: R = R::constant(-0.11) [VOLTAGE],
        /// The channel-length dependence of the threshold's temperature coefficient,
        /// `kt1l`, in volt meters; default 0.
        binned pub kt1l: R = R::constant(0.0) [VOLTAGE.mul(LENGTH)],
        /// The body-coefficient of kt1, `kt2`; default 0.022.
        binned pub kt2: R = R::constant(0.022),
        /// The second-order body effect, `k2`; default derived from `gamma1` and `gamma2`,
        /// or -0.0186 where only `k1` is given.
        binned pub k2: Option<R> = None,
        /// The narrow width effect coefficient, `k3`; default 80.
        binned pub k3: R = R::constant(80.0),
        /// The body effect coefficient of k3, `k3b`, in per volt; default 0.
        binned pub k3b: R = R::constant(0.0) [PER_VOLT],
        /// The narrow width effect parameter, `w0`, in meters; default 2.5e-06.
        binned pub w0: R = R::constant(2.5e-6) [LENGTH],
        /// The first parameter for Vth shift due to pocket, `dvtp0`, in meters; default 0.
        binned pub dvtp0: R = R::constant(0.0) [LENGTH],
        /// The second parameter for Vth shift due to pocket, `dvtp1`, in per volt; default
        /// 0.
        binned pub dvtp1: R = R::constant(0.0) [PER_VOLT],
        /// The equivalent length of pocket region at zero bias, `lpe0`, in meters; default
        /// 1.74e-07.
        binned pub lpe0: R = R::constant(1.74e-7) [LENGTH],
        /// The equivalent length of pocket region accounting for body bias, `lpeb`, in
        /// meters; default 0.
        binned pub lpeb: R = R::constant(0.0) [LENGTH],
        /// The first coefficient of the short-channel effect, `dvt0`; default 2.2.
        binned pub dvt0: R = R::constant(2.2),
        /// The second coefficient of the short-channel effect, `dvt1`; default 0.53.
        binned pub dvt1: R = R::constant(0.53),
        /// The body-bias coefficient of the short-channel effect, `dvt2`, in per volt;
        /// default -0.032.
        binned pub dvt2: R = R::constant(-0.032) [PER_VOLT],
        /// The first coefficient of the narrow-width effect, `dvt0w`, in per meter; default
        /// 0.
        binned pub dvt0w: R = R::constant(0.0) [LENGTH.pow(-1)],
        /// The second coefficient of the narrow-width effect, `dvt1w`, in per meter;
        /// default 5.3e+06.
        binned pub dvt1w: R = R::constant(5.3e6) [LENGTH.pow(-1)],
        /// The body-bias coefficient of the narrow-width effect, `dvt2w`, in per volt;
        /// default -0.032.
        binned pub dvt2w: R = R::constant(-0.032) [PER_VOLT],
        /// The DIBL coefficient of output resistance, `drout`; default 0.56.
        binned pub drout: R = R::constant(0.56),
        /// The DIBL coefficient in the subthreshold region, `dsub`; default `drout`.
        binned pub dsub: Option<R> = None => self.drout,
        /// The zero-bias threshold voltage of a long, wide channel, negative for a
        /// p-channel, `vth0`, in volts; default derived from `vfb`, `phi` and `k1`.
        binned pub vth0: Option<R> = None [VOLTAGE],
        /// The linear gate dependence of mobility, `ua`, in meters per volt; default 1e-9
        /// m/V, ngspice's at `mobmod` 0.
        binned pub ua: Option<R> = None => R::constant(1.0e-9) [LENGTH.div(VOLTAGE)],
        /// The temperature coefficient of ua, `ua1`, in meters per volt; default 1e-09.
        binned pub ua1: R = R::constant(1.0e-9) [LENGTH.div(VOLTAGE)],
        /// The quadratic gate dependence of mobility, `ub`, in square meters per square
        /// volt; default 1e-19.
        binned pub ub: R = R::constant(1.0e-19) [LENGTH.div(VOLTAGE).pow(2)],
        /// The temperature coefficient of ub, `ub1`, in square meters per square volt;
        /// default -1e-18.
        binned pub ub1: R = R::constant(-1.0e-18) [LENGTH.div(VOLTAGE).pow(2)],
        /// The body-bias dependence of mobility, `uc`, in meters per square volt; default
        /// -4.65e-11 m/V², ngspice's at `mobmod` 0.
        binned pub uc: Option<R> = None => R::constant(-0.0465e-9) [LENGTH.div(VOLTAGE.pow(2))],
        /// The temperature coefficient of uc, `uc1`, in meters per square volt; default
        /// -5.6e-11 m/V², ngspice's at `mobmod` 0.
        binned pub uc1: Option<R> = None => R::constant(-0.056e-9) [LENGTH.div(VOLTAGE.pow(2))],
        /// The coulomb scattering factor of mobility, `ud`, in per square meter; default
        /// 1e+14.
        binned pub ud: R = R::constant(1.0e14) [LENGTH.pow(-2)],
        /// The temperature coefficient of ud, `ud1`, in per square meter; default 0.
        binned pub ud1: R = R::constant(0.0) [LENGTH.pow(-2)],
        /// The channel length linear factor of mobility, `up`, in per square meter; default
        /// 0.
        binned pub up: R = R::constant(0.0) [LENGTH.pow(-2)],
        /// The channel length exponential factor of mobility, `lp`, in meters; default
        /// 1e-08.
        binned pub lp: R = R::constant(1.0e-8) [LENGTH],
        /// The low-field mobility at Tnom, `u0`, in square meters per volt-second; default
        /// 0.067 m²/(V·s) for an n-channel and 0.025 m²/(V·s) for a p-channel.
        binned pub u0: Option<R> = None
            => R::constant(by_channel(self.channel, 0.067, 0.025)) [MOBILITY],
        /// The mobility exponent, `eu`; default 1.67 for an n-channel and 1 for a
        /// p-channel.
        binned(card_only) pub eu: Option<R> = None,
        /// The temperature coefficient of mobility, `ute`; default -1.5.
        binned pub ute: R = R::constant(-1.5),
        /// The threshold voltage offset, `voff`, in volts; default -0.08.
        binned pub voff: R = R::constant(-0.08) [VOLTAGE],
        /// The fitting parameter of moderate inversion in Vgsteff, `minv`; default 0.
        binned pub minv: R = R::constant(0.0),
        /// The channel-length dependence of the threshold offset, `voffl`, in volt meters;
        /// default 0.
        pub voffl: R = R::constant(0.0) [VOLTAGE.mul(LENGTH)],
        /// The gate-source overlap capacitance per width, `cgso`, in farads per meter;
        /// default derived from `dlc` and `cgsl`, or from `xj`.
        pub cgso: Option<R> = None [CAPACITANCE.div(LENGTH)],
        /// The gate-drain overlap capacitance per width, `cgdo`, in farads per meter;
        /// default derived from `dlc` and `cgdl`, or from `xj`.
        pub cgdo: Option<R> = None [CAPACITANCE.div(LENGTH)],
        /// The gate-bulk overlap capacitance per length, `cgbo`, in farads per meter;
        /// default twice `dwc` times the oxide capacitance.
        pub cgbo: Option<R> = None [CAPACITANCE.div(LENGTH)],
        /// The effective Vds parameter, `delta`, in volts; default 0.01.
        binned pub delta: R = R::constant(0.01) [VOLTAGE],
        /// The sheet resistance of the source and drain diffusions, `rsh`, in ohms; default
        /// 0.
        pub rsh: R = R::constant(0.0) [RESISTANCE],
        /// The source and drain resistance per width, `rdsw`, in ohms; default 200.
        binned pub rdsw: R = R::constant(200.0) [RESISTANCE],
        /// The source and drain resistance per width at high gate bias, `rdswmin`, in ohms;
        /// default 0.
        pub rdswmin: R = R::constant(0.0) [RESISTANCE],
        /// The source resistance per width, `rsw`, in ohms; default 100.
        binned pub rsw: R = R::constant(100.0) [RESISTANCE],
        /// The drain resistance per width, `rdw`, in ohms; default 100.
        binned pub rdw: R = R::constant(100.0) [RESISTANCE],
        /// The drain resistance per width at high Vg, `rdwmin`, in ohms; default 0.
        pub rdwmin: R = R::constant(0.0) [RESISTANCE],
        /// The source resistance per width at high Vg, `rswmin`, in ohms; default 0.
        pub rswmin: R = R::constant(0.0) [RESISTANCE],
        /// The gate-bias effect on parasitic resistance, `prwg`, in per volt; default 1.
        binned pub prwg: R = R::constant(1.0) [PER_VOLT],
        /// The body-effect on parasitic resistance, `prwb`, in per square-root volt;
        /// default 0.
        binned pub prwb: R = R::constant(0.0) [VOLTAGE.sqrt().recip()],
        /// The temperature coefficient of parasitic resistance, `prt`, in ohms; default 0.
        binned pub prt: R = R::constant(0.0) [RESISTANCE],
        /// The drain-induced barrier lowering in the subthreshold region, `eta0`; default
        /// 0.08.
        binned pub eta0: R = R::constant(0.08),
        /// The body-bias coefficient of `eta0`, `etab`, in per volt; default -0.07.
        binned pub etab: R = R::constant(-0.07) [PER_VOLT],
        /// Channel-length modulation coefficient, `pclm`; default 1.3.
        binned pub pclm: R = R::constant(1.3),
        /// The first coefficient of the drain-induced barrier lowering of the output
        /// resistance, `pdiblc1`; default 0.39.
        binned pub pdiblc1: R = R::constant(0.39),
        /// The second coefficient of the drain-induced barrier lowering of the output
        /// resistance, `pdiblc2`; default 0.0086.
        binned pub pdiblc2: R = R::constant(0.0086),
        /// The body-effect on drain-induced barrier lowering, `pdiblcb`, in per volt;
        /// default 0.
        binned pub pdiblcb: R = R::constant(0.0) [PER_VOLT],
        /// The rout degradation coefficient for pocket devices, `fprout`, in volts per
        /// square-root meter; default 0.
        binned pub fprout: R = R::constant(0.0) [VOLTAGE.div(LENGTH.sqrt())],
        /// The coefficient for drain-induced Vth shifts, `pdits`, in per volt; default 0.
        binned pub pdits: R = R::constant(0.0) [PER_VOLT],
        /// The length dependence of drain-induced Vth shifts, `pditsl`, in per meter;
        /// default 0.
        pub pditsl: R = R::constant(0.0) [LENGTH.pow(-1)],
        /// The vds dependence of drain-induced Vth shifts, `pditsd`, in per volt; default
        /// 0.
        binned pub pditsd: R = R::constant(0.0) [PER_VOLT],
        /// The exponential coefficient of the substrate current's effect on the output
        /// resistance, `pscbe1`, in volts per meter; default 4.24e+08.
        binned pub pscbe1: R = R::constant(4.24e8) [VOLTAGE.div(LENGTH)],
        /// The coefficient of the substrate current's effect on the output resistance,
        /// `pscbe2`, in meters per volt; default 1e-05.
        binned pub pscbe2: R = R::constant(1.0e-5) [LENGTH.div(VOLTAGE)],
        /// The gate dependence of output resistance parameter, `pvag`; default 0.
        binned pub pvag: R = R::constant(0.0),
        /// The bottom source junction reverse saturation current density, `jss`, in amperes
        /// per square meter; default 0.0001.
        pub jss: R = R::constant(0.0001) [CURRENT.div(LENGTH.pow(2))],
        /// The isolation edge sidewall source junction reverse saturation current density,
        /// `jsws`, in amperes per meter; default 0.
        pub jsws: R = R::constant(0.0) [CURRENT.div(LENGTH)],
        /// The gate edge source junction reverse saturation current density, `jswgs`, in
        /// amperes per meter; default 0.
        pub jswgs: R = R::constant(0.0) [CURRENT.div(LENGTH)],
        /// The source junction built-in potential, `pbs`, in volts; default 1.
        pub pbs: R = R::constant(1.0) [VOLTAGE],
        /// The source junction emission coefficient, `njs`; default 1.
        pub njs: R = R::constant(1.0),
        /// The source junction current temperature exponent, `xtis`; default 3.
        pub xtis: R = R::constant(3.0),
        /// The source bottom junction capacitance grading coefficient, `mjs`; default 0.5.
        pub mjs: R = R::constant(0.5),
        /// The source sidewall junction capacitance built in potential, `pbsws`, in volts;
        /// default 1.
        pub pbsws: R = R::constant(1.0) [VOLTAGE],
        /// The source sidewall junction capacitance grading coefficient, `mjsws`; default
        /// 0.33.
        pub mjsws: R = R::constant(0.33),
        /// The source (gate side) sidewall junction capacitance built in potential,
        /// `pbswgs`, in volts; default `pbsws`.
        pub pbswgs: Option<R> = None [VOLTAGE],
        /// The source (gate side) sidewall junction capacitance grading coefficient,
        /// `mjswgs`; default `mjsws`.
        pub mjswgs: Option<R> = None,
        /// The source bottom junction capacitance per unit area, `cjs`, in farads per
        /// square meter; default 0.0005.
        pub cjs: R = R::constant(0.0005) [CAPACITANCE.div(LENGTH.pow(2))],
        /// The source sidewall junction capacitance per unit periphery, `cjsws`, in farads
        /// per meter; default 5e-10.
        pub cjsws: R = R::constant(5.0e-10) [CAPACITANCE.div(LENGTH)],
        /// The source (gate side) sidewall junction capacitance per unit width, `cjswgs`,
        /// in farads per meter; default `cjsws`.
        pub cjswgs: Option<R> = None [CAPACITANCE.div(LENGTH)],
        /// The bottom drain junction reverse saturation current density, `jsd`, in amperes
        /// per square meter; default `jss`.
        pub jsd: Option<R> = None [CURRENT.div(LENGTH.pow(2))],
        /// The isolation edge sidewall drain junction reverse saturation current density,
        /// `jswd`, in amperes per meter; default `jsws`.
        pub jswd: Option<R> = None [CURRENT.div(LENGTH)],
        /// The gate edge drain junction reverse saturation current density, `jswgd`, in
        /// amperes per meter; default `jswgs`.
        pub jswgd: Option<R> = None [CURRENT.div(LENGTH)],
        /// The drain junction built-in potential, `pbd`, in volts; default `pbs`.
        pub pbd: Option<R> = None [VOLTAGE],
        /// The drain junction emission coefficient, `njd`; default `njs`.
        pub njd: Option<R> = None,
        /// The drain junction current temperature exponent, `xtid`; default `xtis`.
        pub xtid: Option<R> = None,
        /// The drain bottom junction capacitance grading coefficient, `mjd`; default `mjs`.
        pub mjd: Option<R> = None,
        /// The drain sidewall junction capacitance built in potential, `pbswd`, in volts;
        /// default `pbsws`.
        pub pbswd: Option<R> = None [VOLTAGE],
        /// The drain sidewall junction capacitance grading coefficient, `mjswd`; default
        /// `mjsws`.
        pub mjswd: Option<R> = None,
        /// The drain (gate side) sidewall junction capacitance built in potential,
        /// `pbswgd`, in volts; default `pbswgs`.
        pub pbswgd: Option<R> = None [VOLTAGE],
        /// The drain (gate side) sidewall junction capacitance grading coefficient,
        /// `mjswgd`; default `mjswgs`.
        pub mjswgd: Option<R> = None,
        /// The drain bottom junction capacitance per unit area, `cjd`, in farads per square
        /// meter; default `cjs`.
        pub cjd: Option<R> = None [CAPACITANCE.div(LENGTH.pow(2))],
        /// The drain sidewall junction capacitance per unit periphery, `cjswd`, in farads
        /// per meter; default `cjsws`.
        pub cjswd: Option<R> = None [CAPACITANCE.div(LENGTH)],
        /// The drain (gate side) sidewall junction capacitance per unit width, `cjswgd`, in
        /// farads per meter; default `cjswgs`.
        pub cjswgd: Option<R> = None [CAPACITANCE.div(LENGTH)],
        /// The flat-band voltage of the charges at `capmod` 0, `vfbcv`, in volts; default
        /// -1.
        binned(card_only) pub vfbcv: R = R::constant(-1.0) [VOLTAGE],
        /// The flat-band voltage, `vfb`, in volts; default derived from `vth0`, or -1 V.
        binned pub vfb: Option<R> = None [VOLTAGE],
        /// The temperature coefficient of pb, `tpb`, in volts per kelvin; default 0.
        pub tpb: R = R::constant(0.0) [VOLTS_PER_KELVIN],
        /// The temperature coefficient of cj, `tcj`, in per kelvin; default 0.
        pub tcj: R = R::constant(0.0) [PER_KELVIN],
        /// The temperature coefficient of pbsw, `tpbsw`, in volts per kelvin; default 0.
        pub tpbsw: R = R::constant(0.0) [VOLTS_PER_KELVIN],
        /// The temperature coefficient of cjsw, `tcjsw`, in per kelvin; default 0.
        pub tcjsw: R = R::constant(0.0) [PER_KELVIN],
        /// The temperature coefficient of pbswg, `tpbswg`, in volts per kelvin; default 0.
        pub tpbswg: R = R::constant(0.0) [VOLTS_PER_KELVIN],
        /// The temperature coefficient of cjswg, `tcjswg`, in per kelvin; default 0.
        pub tcjswg: R = R::constant(0.0) [PER_KELVIN],
        /// The exponential coefficient for finite charge thickness, `acde`, in meters per
        /// volt; default 1.
        binned pub acde: R = R::constant(1.0) [LENGTH.div(VOLTAGE)],
        /// The coefficient for gate-bias dependent surface potential, `moin`; default 15.
        binned pub moin: R = R::constant(15.0),
        /// The C-V turn-on/off parameter, `noff`; default 1.
        binned pub noff: R = R::constant(1.0),
        /// The C-V lateral-shift parameter, `voffcv`, in volts; default 0.
        binned pub voffcv: R = R::constant(0.0) [VOLTAGE],
        /// The distance of Mid-Contact to Gate edge, `dmcg`, in meters; default 0.
        pub dmcg: R = R::constant(0.0) [LENGTH],
        /// The distance of Mid-Contact to Isolation, `dmci`, in meters; default `dmcg`.
        pub dmci: Option<R> = None [LENGTH],
        /// The distance of Mid-Diffusion to Gate edge, `dmdg`, in meters; default 0.
        pub dmdg: R = R::constant(0.0) [LENGTH],
        /// The distance of Mid-Contact to Gate edge in Test structures, `dmcgt`, in meters;
        /// default 0.
        pub dmcgt: R = R::constant(0.0) [LENGTH],
        /// The distance from gate contact center to device edge, `xgw`, in meters; default
        /// 0.
        pub xgw: R = R::constant(0.0) [LENGTH],
        /// The offset of the gate's drawn length, `xgl`, in meters; default 0.
        pub xgl: R = R::constant(0.0) [LENGTH],
        /// The gate sheet resistance, `rshg`, in ohms; default 0.1.
        pub rshg: R = R::constant(0.1) [RESISTANCE],
        /// The number of gate contacts, `ngcon`; default 1.
        pub ngcon: R = R::constant(1.0),
        /// The first fitting parameter of the bias-dependent gate resistance, `xrcrg1`;
        /// default 12.
        binned pub xrcrg1: R = R::constant(12.0),
        /// The second fitting parameter of the bias-dependent gate resistance, `xrcrg2`;
        /// default 1.
        binned pub xrcrg2: R = R::constant(1.0),
        /// The velocity overshoot parameter, `lambda`; default 0, and no velocity overshoot
        /// where it is not given.
        binned pub lambda: Option<R> = None => R::constant(0.0),
        /// The thermal velocity, `vtl`, in meters per second; default 2e5 m/s, and no limit
        /// of the source end's velocity where it is not given.
        binned pub vtl: Option<R> = None => R::constant(2.0e5) [LENGTH.div(TIME)],
        /// The backscattering mean free path, `lc`, in meters; default 5e-09.
        pub lc: R = R::constant(5.0e-9) [LENGTH],
        /// The backscattering coefficient, `xn`; default 3.
        binned pub xn: R = R::constant(3.0),
        /// The S/D flatband voltage offset, `vfbsdoff`, in volts; default 0.
        binned(card_only) pub vfbsdoff: R = R::constant(0.0) [VOLTAGE],
        /// The temperature parameter for vfbsdoff, `tvfbsdoff`, in per kelvin; default 0.
        binned(card_only) pub tvfbsdoff: R = R::constant(0.0) [PER_KELVIN],
        /// The temperature parameter for voff, `tvoff`, in per kelvin; default 0.
        binned pub tvoff: R = R::constant(0.0) [PER_KELVIN],
        /// The length each side of the channel loses in the flicker noise, beyond `lint`,
        /// `lintnoi`, in meters; default 0.
        pub lintnoi: R = R::constant(0.0) [LENGTH],
        /// The length lost on each side of the channel, `lint`, in meters; default 0.
        pub lint: R = R::constant(0.0) [LENGTH],
        /// The coefficient of the length dependence of the length lost, `ll`, in the
        /// manual's units; default 0.
        pub ll: R = R::constant(0.0),
        /// The coefficient of the length dependence of the length lost in the charges,
        /// `llc`, in the manual's units; default `ll`.
        pub llc: Option<R> = None,
        /// The exponent of the length dependence of the length lost, `lln`; default 1.
        pub lln: R = R::constant(1.0),
        /// The coefficient of the width dependence of the length lost, `lw`, in the
        /// manual's units; default 0.
        pub lw: R = R::constant(0.0),
        /// The coefficient of the width dependence of the length lost in the charges,
        /// `lwc`, in the manual's units; default `lw`.
        pub lwc: Option<R> = None,
        /// The exponent of the width dependence of the length lost, `lwn`; default 1.
        pub lwn: R = R::constant(1.0),
        /// The coefficient of the area dependence of the length lost, `lwl`, in the
        /// manual's units; default 0.
        pub lwl: R = R::constant(0.0),
        /// The coefficient of the area dependence of the length lost in the charges,
        /// `lwlc`, in the manual's units; default `lwl`.
        pub lwlc: Option<R> = None,
        /// The width exponent of the source and drain resistance, `wr`; default 1.
        binned pub wr: R = R::constant(1.0),
        /// The width lost on each side of the channel, `wint`, in meters; default 0.
        pub wint: R = R::constant(0.0) [LENGTH],
        /// The gate-bias dependence of the width lost, `dwg`, in meters per volt; default
        /// 0.
        binned pub dwg: R = R::constant(0.0) [LENGTH.div(VOLTAGE)],
        /// The body-bias dependence of the width lost, `dwb`, in meters per square-root
        /// volt; default 0.
        binned pub dwb: R = R::constant(0.0) [LENGTH.div(VOLTAGE.sqrt())],
        /// The coefficient of the length dependence of the width lost, `wl`, in the
        /// manual's units; default 0.
        pub wl: R = R::constant(0.0),
        /// The coefficient of the length dependence of the width lost in the charges,
        /// `wlc`, in the manual's units; default `wl`.
        pub wlc: Option<R> = None,
        /// The exponent of the length dependence of the width lost, `wln`; default 1.
        pub wln: R = R::constant(1.0),
        /// The coefficient of the width dependence of the width lost, `ww`, in the manual's
        /// units; default 0.
        pub ww: R = R::constant(0.0),
        /// The coefficient of the width dependence of the width lost in the charges, `wwc`,
        /// in the manual's units; default `ww`.
        pub wwc: Option<R> = None,
        /// The exponent of the width dependence of the width lost, `wwn`; default 1.
        pub wwn: R = R::constant(1.0),
        /// The coefficient of the area dependence of the width lost, `wwl`, in the manual's
        /// units; default 0.
        pub wwl: R = R::constant(0.0),
        /// The coefficient of the area dependence of the width lost in the charges, `wwlc`,
        /// in the manual's units; default `wwl`.
        pub wwlc: Option<R> = None,
        /// The first narrow-width coefficient of Abulk, `b0`, in meters; default 0.
        binned pub b0: R = R::constant(0.0) [LENGTH],
        /// The second narrow-width coefficient of Abulk, `b1`, in meters; default 0.
        binned pub b1: R = R::constant(0.0) [LENGTH],
        /// The gate-source overlap capacitance per width of the lightly doped extension,
        /// `cgsl`, in farads per meter; default 0.
        binned pub cgsl: R = R::constant(0.0) [CAPACITANCE.div(LENGTH)],
        /// The gate-drain overlap capacitance per width of the lightly doped extension,
        /// `cgdl`, in farads per meter; default 0.
        binned pub cgdl: R = R::constant(0.0) [CAPACITANCE.div(LENGTH)],
        /// The coefficient of the bias dependence of the gate-source overlap, `ckappas`, in
        /// volts; default 0.6.
        binned pub ckappas: R = R::constant(0.6) [VOLTAGE],
        /// The coefficient of the bias dependence of the gate-drain overlap, `ckappad`, in
        /// volts; default `ckappas`.
        binned pub ckappad: Option<R> = None => self.ckappas [VOLTAGE],
        /// Fringe capacitance parameter, `cf`, in farads per meter; default
        /// `2 · epsrox · ε0 / π · ln(1 + 0.4 µm / toxe)` using `toxe` when given or 3 nm.
        binned pub cf: Option<R> = None
            => fringe(self.epsrox, self.set_toxe()) [CAPACITANCE.div(LENGTH)],
        /// The vdsat parameter for C-V model, `clc`, in meters; default 1e-07.
        binned pub clc: R = R::constant(1.0e-7) [LENGTH],
        /// The vdsat parameter for C-V model, `cle`; default 0.6.
        binned pub cle: R = R::constant(0.6),
        /// The width lost on each side of the channel in the charges, `dwc`, in meters;
        /// default `wint`.
        pub dwc: Option<R> = None [LENGTH],
        /// The length lost on each side of the channel in the charges, `dlc`, in meters;
        /// default `lint`.
        pub dlc: Option<R> = None [LENGTH],
        /// The offset of the channel's width from masking and etching, `xw`, in meters;
        /// default 0.
        pub xw: R = R::constant(0.0) [LENGTH],
        /// The offset of the channel's length from masking and etching, `xl`, in meters;
        /// default 0.
        pub xl: R = R::constant(0.0) [LENGTH],
        /// The length lost on each side of the channel in the gate tunneling currents,
        /// `dlcig`, in meters; default `lint`.
        pub dlcig: Option<R> = None [LENGTH],
        /// The width lost on each side of the source and drain junctions, `dwj`, in meters;
        /// default `dwc`.
        pub dwj: Option<R> = None [LENGTH],
        /// The first coefficient of the impact ionization's substrate current, `alpha0`, in
        /// meters per volt; default 0.
        binned pub alpha0: R = R::constant(0.0) [LENGTH.div(VOLTAGE)],
        /// The channel-length dependence of the substrate current, `alpha1`, in per volt;
        /// default 0.
        binned pub alpha1: R = R::constant(0.0) [PER_VOLT],
        /// The exponential coefficient of the substrate current, `beta0`, in volts; default
        /// 0.
        binned pub beta0: R = R::constant(0.0) [VOLTAGE],
        /// The pre-exponential constant for GIDL, `agidl`, in siemens; default 0.
        binned pub agidl: R = R::constant(0.0) [CONDUCTANCE],
        /// The exponential constant for GIDL, `bgidl`, in volts per meter; default 2.3e+09.
        binned pub bgidl: R = R::constant(2.3e9) [VOLTAGE.div(LENGTH)],
        /// The parameter for body-bias dependence of GIDL, `cgidl`, in cubic volts; default
        /// 0.5.
        binned pub cgidl: R = R::constant(0.5) [VOLTAGE.pow(3)],
        /// The band bending of the gate-induced drain leakage, `egidl`, in volts; default
        /// 0.8.
        binned pub egidl: R = R::constant(0.8) [VOLTAGE],
        /// The parameter for Igc, `aigc`, in the manual's units; default 1.36e-2 for an
        /// n-channel and 9.8e-3 for a p-channel.
        binned(card_only) pub aigc: Option<R> = None,
        /// The parameter for Igc, `bigc`, in the manual's units; default 1.71e-3 for an
        /// n-channel and 7.59e-4 for a p-channel.
        binned(card_only) pub bigc: Option<R> = None,
        /// The parameter for Igc, `cigc`, in per volt; default 0.075 per volt for an
        /// n-channel and 0.03 per volt for a p-channel.
        binned(card_only) pub cigc: Option<R> = None [PER_VOLT],
        /// The parameter for Igs,d, `aigsd`, in the manual's units; default 1.36e-2 for an
        /// n-channel and 9.8e-3 for a p-channel.
        binned(card_only) pub aigsd: Option<R> = None,
        /// The parameter for Igs,d, `bigsd`, in the manual's units; default 1.71e-3 for an
        /// n-channel and 7.59e-4 for a p-channel.
        binned(card_only) pub bigsd: Option<R> = None,
        /// The parameter for Igs,d, `cigsd`, in per volt; default 0.075 per volt for an
        /// n-channel and 0.03 per volt for a p-channel.
        binned(card_only) pub cigsd: Option<R> = None [PER_VOLT],
        /// The parameter for Igb, `aigbacc`, in the manual's units; default 0.0136.
        binned(card_only) pub aigbacc: R = R::constant(0.0136),
        /// The parameter for Igb, `bigbacc`, in the manual's units; default 0.00171.
        binned(card_only) pub bigbacc: R = R::constant(1.71e-3),
        /// The parameter for Igb, `cigbacc`, in per volt; default 0.075.
        binned(card_only) pub cigbacc: R = R::constant(0.075) [PER_VOLT],
        /// The parameter for Igb, `aigbinv`, in the manual's units; default 0.0111.
        binned(card_only) pub aigbinv: R = R::constant(0.0111),
        /// The parameter for Igb, `bigbinv`, in the manual's units; default 0.000949.
        binned(card_only) pub bigbinv: R = R::constant(9.49e-4),
        /// The parameter for Igb, `cigbinv`, in per volt; default 0.006.
        binned(card_only) pub cigbinv: R = R::constant(0.006) [PER_VOLT],
        /// The parameter for Igc slope, `nigc`; default 1.
        binned(card_only) pub nigc: R = R::constant(1.0),
        /// The parameter for Igbinv slope, `nigbinv`; default 3.
        binned(card_only) pub nigbinv: R = R::constant(3.0),
        /// The parameter for Igbacc slope, `nigbacc`; default 1.
        binned(card_only) pub nigbacc: R = R::constant(1.0),
        /// The exponent for Tox ratio, `ntox`; default 1.
        binned(card_only) pub ntox: R = R::constant(1.0),
        /// The parameter for the Si bandgap for Igbinv, `eigbinv`, in volts; default 1.1.
        binned(card_only) pub eigbinv: R = R::constant(1.1) [VOLTAGE],
        /// The parameter for Igc partition, `pigcd`; default 1.
        binned(card_only) pub pigcd: R = R::constant(1.0),
        /// The factor for the gate edge Tox, `poxedge`; default 1.
        binned(card_only) pub poxedge: R = R::constant(1.0),
        /// The forward drain diode forward limiting current, `ijthdfwd`, in amperes;
        /// default `ijthsfwd`.
        pub ijthdfwd: Option<R> = None [CURRENT],
        /// The forward source diode forward limiting current, `ijthsfwd`, in amperes;
        /// default 0.1.
        pub ijthsfwd: R = R::constant(0.1) [CURRENT],
        /// The reverse drain diode forward limiting current, `ijthdrev`, in amperes;
        /// default `ijthsrev`.
        pub ijthdrev: Option<R> = None [CURRENT],
        /// The reverse source diode forward limiting current, `ijthsrev`, in amperes;
        /// default 0.1.
        pub ijthsrev: R = R::constant(0.1) [CURRENT],
        /// The fitting parameter for drain diode breakdown current, `xjbvd`; default
        /// `xjbvs`.
        pub xjbvd: Option<R> = None,
        /// The fitting parameter for source diode breakdown current, `xjbvs`; default 1.
        pub xjbvs: R = R::constant(1.0),
        /// The drain diode breakdown voltage, `bvd`, in volts; default `bvs`.
        pub bvd: Option<R> = None [VOLTAGE],
        /// The source diode breakdown voltage, `bvs`, in volts; default 10.
        pub bvs: R = R::constant(10.0) [VOLTAGE],
        /// The source bottom trap-assisted saturation current density, `jtss`, in amperes
        /// per square meter; default 0.
        pub jtss: R = R::constant(0.0) [CURRENT.div(LENGTH.pow(2))],
        /// The drain bottom trap-assisted saturation current density, `jtsd`, in amperes
        /// per square meter; default `jtss`.
        pub jtsd: Option<R> = None [CURRENT.div(LENGTH.pow(2))],
        /// The source STI sidewall trap-assisted saturation current density, `jtssws`, in
        /// amperes per meter; default 0.
        pub jtssws: R = R::constant(0.0) [CURRENT.div(LENGTH)],
        /// The drain STI sidewall trap-assisted saturation current density, `jtsswd`, in
        /// amperes per meter; default `jtssws`.
        pub jtsswd: Option<R> = None [CURRENT.div(LENGTH)],
        /// The source gate-edge sidewall trap-assisted saturation current density,
        /// `jtsswgs`, in amperes per meter; default 0.
        pub jtsswgs: R = R::constant(0.0) [CURRENT.div(LENGTH)],
        /// The drain gate-edge sidewall trap-assisted saturation current density,
        /// `jtsswgd`, in amperes per meter; default `jtsswgs`.
        pub jtsswgd: Option<R> = None [CURRENT.div(LENGTH)],
        /// The non-ideality factor for bottom junction, `njts`; default 20.
        pub njts: R = R::constant(20.0),
        /// The non-ideality factor for STI sidewall junction, `njtssw`; default 20.
        pub njtssw: R = R::constant(20.0),
        /// The non-ideality factor for gate-edge sidewall junction, `njtsswg`; default 20.
        pub njtsswg: R = R::constant(20.0),
        /// The power dependence of JTSS on temperature, `xtss`; default 0.02.
        pub xtss: R = R::constant(0.02),
        /// The power dependence of JTSD on temperature, `xtsd`; default `xtss`.
        pub xtsd: Option<R> = None,
        /// The power dependence of JTSSWS on temperature, `xtssws`; default 0.02.
        pub xtssws: R = R::constant(0.02),
        /// The power dependence of JTSSWD on temperature, `xtsswd`; default `xtssws`.
        pub xtsswd: Option<R> = None,
        /// The power dependence of JTSSWGS on temperature, `xtsswgs`; default 0.02.
        pub xtsswgs: R = R::constant(0.02),
        /// The power dependence of JTSSWGD on temperature, `xtsswgd`; default `xtsswgs`.
        pub xtsswgd: Option<R> = None,
        /// The temperature coefficient for NJTS, `tnjts`; default 0.
        pub tnjts: R = R::constant(0.0),
        /// The temperature coefficient for NJTSSW, `tnjtssw`; default 0.
        pub tnjtssw: R = R::constant(0.0),
        /// The temperature coefficient for NJTSSWG, `tnjtsswg`; default 0.
        pub tnjtsswg: R = R::constant(0.0),
        /// The source bottom trap-assisted voltage dependent parameter, `vtss`, in volts;
        /// default 10.
        pub vtss: R = R::constant(10.0) [VOLTAGE],
        /// The drain bottom trap-assisted voltage dependent parameter, `vtsd`, in volts;
        /// default `vtss`.
        pub vtsd: Option<R> = None [VOLTAGE],
        /// The source STI sidewall trap-assisted voltage dependent parameter, `vtssws`, in
        /// volts; default 10.
        pub vtssws: R = R::constant(10.0) [VOLTAGE],
        /// The drain STI sidewall trap-assisted voltage dependent parameter, `vtsswd`, in
        /// volts; default `vtssws`.
        pub vtsswd: Option<R> = None [VOLTAGE],
        /// The source gate-edge sidewall trap-assisted voltage dependent parameter,
        /// `vtsswgs`, in volts; default 10.
        pub vtsswgs: R = R::constant(10.0) [VOLTAGE],
        /// The drain gate-edge sidewall trap-assisted voltage dependent parameter,
        /// `vtsswgd`, in volts; default `vtsswgs`.
        pub vtsswgd: Option<R> = None [VOLTAGE],
        /// The least conductance of each body resistance, `gbmin`, in siemens; default
        /// 1e-12.
        pub gbmin: R = R::constant(1.0e-12) [CONDUCTANCE],
        /// The resistance from the drain's body node to the bulk, `rbdb`, in ohms; default
        /// 50.
        pub rbdb: R = R::constant(50.0) [RESISTANCE],
        /// The resistance from the channel's body node to the bulk, `rbpb`, in ohms;
        /// default 50.
        pub rbpb: R = R::constant(50.0) [RESISTANCE],
        /// The resistance from the source's body node to the bulk, `rbsb`, in ohms; default
        /// 50.
        pub rbsb: R = R::constant(50.0) [RESISTANCE],
        /// The resistance from the channel's body node to the source's, `rbps`, in ohms;
        /// default 50.
        pub rbps: R = R::constant(50.0) [RESISTANCE],
        /// The resistance from the channel's body node to the drain's, `rbpd`, in ohms;
        /// default 50.
        pub rbpd: R = R::constant(50.0) [RESISTANCE],
        /// The scaling of `rbps` at `rbodymod` 2, `rbps0`, in ohms; default 50.
        pub rbps0: R = R::constant(50.0) [RESISTANCE],
        /// The body resistance RBPS L scaling, `rbpsl`; default 0.
        pub rbpsl: R = R::constant(0.0),
        /// The body resistance RBPS W scaling, `rbpsw`; default 0.
        pub rbpsw: R = R::constant(0.0),
        /// The body resistance RBPS NF scaling, `rbpsnf`; default 0.
        pub rbpsnf: R = R::constant(0.0),
        /// The scaling of `rbpd` at `rbodymod` 2, `rbpd0`, in ohms; default 50.
        pub rbpd0: R = R::constant(50.0) [RESISTANCE],
        /// The body resistance RBPD L scaling, `rbpdl`; default 0.
        pub rbpdl: R = R::constant(0.0),
        /// The body resistance RBPD W scaling, `rbpdw`; default 0.
        pub rbpdw: R = R::constant(0.0),
        /// The body resistance RBPD NF scaling, `rbpdnf`; default 0.
        pub rbpdnf: R = R::constant(0.0),
        /// The scaling of `rbpbx` at `rbodymod` 2, `rbpbx0`, in ohms; default 100.
        pub rbpbx0: R = R::constant(100.0) [RESISTANCE],
        /// The body resistance RBPBX L scaling, `rbpbxl`; default 0.
        pub rbpbxl: R = R::constant(0.0),
        /// The body resistance RBPBX W scaling, `rbpbxw`; default 0.
        pub rbpbxw: R = R::constant(0.0),
        /// The body resistance RBPBX NF scaling, `rbpbxnf`; default 0.
        pub rbpbxnf: R = R::constant(0.0),
        /// The scaling of `rbpby` at `rbodymod` 2, `rbpby0`, in ohms; default 100.
        pub rbpby0: R = R::constant(100.0) [RESISTANCE],
        /// The body resistance RBPBY L scaling, `rbpbyl`; default 0.
        pub rbpbyl: R = R::constant(0.0),
        /// The body resistance RBPBY W scaling, `rbpbyw`; default 0.
        pub rbpbyw: R = R::constant(0.0),
        /// The body resistance RBPBY NF scaling, `rbpbynf`; default 0.
        pub rbpbynf: R = R::constant(0.0),
        /// The scaling of `rbsbx` at `rbodymod` 2, `rbsbx0`, in ohms; default 100.
        pub rbsbx0: R = R::constant(100.0) [RESISTANCE],
        /// The scaling of `rbsby` at `rbodymod` 2, `rbsby0`, in ohms; default 100.
        pub rbsby0: R = R::constant(100.0) [RESISTANCE],
        /// The scaling of `rbdbx` at `rbodymod` 2, `rbdbx0`, in ohms; default 100.
        pub rbdbx0: R = R::constant(100.0) [RESISTANCE],
        /// The scaling of `rbdby` at `rbodymod` 2, `rbdby0`, in ohms; default 100.
        pub rbdby0: R = R::constant(100.0) [RESISTANCE],
        /// The body resistance RBSDBX L scaling, `rbsdbxl`; default 0.
        pub rbsdbxl: R = R::constant(0.0),
        /// The body resistance RBSDBX W scaling, `rbsdbxw`; default 0.
        pub rbsdbxw: R = R::constant(0.0),
        /// The body resistance RBSDBX NF scaling, `rbsdbxnf`; default 0.
        pub rbsdbxnf: R = R::constant(0.0),
        /// The body resistance RBSDBY L scaling, `rbsdbyl`; default 0.
        pub rbsdbyl: R = R::constant(0.0),
        /// The body resistance RBSDBY W scaling, `rbsdbyw`; default 0.
        pub rbsdbyw: R = R::constant(0.0),
        /// The body resistance RBSDBY NF scaling, `rbsdbynf`; default 0.
        pub rbsdbynf: R = R::constant(0.0),
        /// The reference distance between OD edge to poly of one side, `saref`, in meters;
        /// default 1e-06.
        pub saref: R = R::constant(1.0e-6) [LENGTH],
        /// The reference distance between OD edge to poly of the other side, `sbref`, in
        /// meters; default 1e-06.
        pub sbref: R = R::constant(1.0e-6) [LENGTH],
        /// The width parameter for stress effect, `wlod`, in meters; default 0.
        pub wlod: R = R::constant(0.0) [LENGTH],
        /// The mobility degradation/enhancement coefficient for LOD, `ku0`, in meters;
        /// default 0.
        pub ku0: R = R::constant(0.0) [LENGTH],
        /// The saturation velocity degradation/enhancement parameter for LOD, `kvsat`;
        /// default 0.
        pub kvsat: R = R::constant(0.0),
        /// The threshold degradation/enhancement parameter for LOD, `kvth0`, in volt
        /// meters; default 0.
        pub kvth0: R = R::constant(0.0) [VOLTAGE.mul(LENGTH)],
        /// The temperature coefficient of KU0, `tku0`; default 0.
        pub tku0: R = R::constant(0.0),
        /// The length parameter for u0 LOD effect, `llodku0`; default 0.
        pub llodku0: R = R::constant(0.0),
        /// The width parameter for u0 LOD effect, `wlodku0`; default 0.
        pub wlodku0: R = R::constant(0.0),
        /// The length parameter for vth LOD effect, `llodvth`; default 0.
        pub llodvth: R = R::constant(0.0),
        /// The width parameter for vth LOD effect, `wlodvth`; default 0.
        pub wlodvth: R = R::constant(0.0),
        /// The length dependence of ku0, `lku0`, in the manual's units; default 0.
        pub lku0: R = R::constant(0.0),
        /// The width dependence of ku0, `wku0`, in the manual's units; default 0.
        pub wku0: R = R::constant(0.0),
        /// The cross-term dependence of ku0, `pku0`, in the manual's units; default 0.
        pub pku0: R = R::constant(0.0),
        /// The length dependence of kvth0, `lkvth0`, in the manual's units; default 0.
        pub lkvth0: R = R::constant(0.0),
        /// The width dependence of kvth0, `wkvth0`, in the manual's units; default 0.
        pub wkvth0: R = R::constant(0.0),
        /// The cross-term dependence of kvth0, `pkvth0`, in the manual's units; default 0.
        pub pkvth0: R = R::constant(0.0),
        /// The K2 shift factor related to stress effect on vth, `stk2`, in meters; default
        /// 0.
        pub stk2: R = R::constant(0.0) [LENGTH],
        /// The K2 shift modification factor for stress effect, `lodk2`; default 1.
        pub lodk2: R = R::constant(1.0),
        /// The eta0 shift factor related to stress effect on vth, `steta0`, in meters;
        /// default 0.
        pub steta0: R = R::constant(0.0) [LENGTH],
        /// The eta0 shift modification factor for stress effect, `lodeta0`; default 1.
        pub lodeta0: R = R::constant(1.0),
        /// The coefficient for SCB, `web`; default 0.
        pub web: R = R::constant(0.0),
        /// The coefficient for SCC, `wec`; default 0.
        pub wec: R = R::constant(0.0),
        /// The threshold shift factor for well proximity effect, `kvth0we`, in volts;
        /// default 0.
        binned(card_only) pub kvth0we: R = R::constant(0.0) [VOLTAGE],
        /// The K2 shift factor for well proximity effect, `k2we`; default 0.
        binned(card_only) pub k2we: R = R::constant(0.0),
        /// The mobility degradation factor for well proximity effect, `ku0we`; default 0.
        binned(card_only) pub ku0we: R = R::constant(0.0),
        /// The reference distance to calculate SCA, SCB and SCC, `scref`, in meters;
        /// default 1e-06.
        pub scref: R = R::constant(1.0e-6) [LENGTH],
        /// The flicker noise parameter, `noia`, in the manual's units; default 6.25e41 for
        /// an n-channel and 6.188e40 for a p-channel.
        pub noia: Option<R> = None,
        /// The flicker noise parameter, `noib`, in the manual's units; default 3.125e26 for
        /// an n-channel and 1.5e25 for a p-channel.
        pub noib: Option<R> = None,
        /// The flicker noise parameter, `noic`, in the manual's units; default 8.75e+09.
        pub noic: R = R::constant(8.75e9),
        /// The thermal noise parameter, `tnoia`; default 1.5.
        pub tnoia: R = R::constant(1.5),
        /// The thermal noise parameter, `tnoib`; default 3.5.
        pub tnoib: R = R::constant(3.5),
        /// The thermal noise coefficient, `rnoia`; default 0.577.
        pub rnoia: R = R::constant(0.577),
        /// The thermal noise coefficient, `rnoib`; default 0.5164.
        pub rnoib: R = R::constant(0.5164),
        /// The thermal noise parameter, `ntnoi`; default 1.
        pub ntnoi: R = R::constant(1.0),
        /// The flicker noise parameter, `em`, in volts per meter; default 4.1e+07.
        pub em: R = R::constant(4.1e7) [VOLTAGE.div(LENGTH)],
        /// The flicker noise frequency exponent, `ef`; default 1.
        pub ef: R = R::constant(1.0),
        /// The flicker noise exponent, `af`; default 1.
        pub af: R = R::constant(1.0),
        /// The flicker noise coefficient, `kf`, in the manual's units; default 0.
        pub kf: R = R::constant(0.0),
    }

    /// The binned parameters the modes jolty models read, at one size: each the card's value
    /// plus its length, width and area dependences at the reciprocals of the effective length,
    /// width and their product.
    #[derive(Debug, Clone, Copy)]
    pub(super) struct Binned<R: Real>;
}
