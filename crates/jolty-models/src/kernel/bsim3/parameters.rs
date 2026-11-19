//! The parameters of a BSIM3 transistor, by the names of the BSIM3v3.3 manual, in SI
//! units, with ngspice's defaults.
//!
//! Defaults match ngspice 47 `b3set.c`. Parameters whose defaults derive from other
//! fields (such as `vth0` from `vfb`) are optional, and each field description states
//! its derivation rule.
//!
//! Part of jolty's BSIM3, derived from BSIM3v3.3.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2004 Regents of the University of California. All rights reserved. Used
//! under the BSIM terms of use, which the repository's `NOTICE` reproduces.

use jolty_derive::binned_parameters;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{Charges, Mobility, NoiseModel, by_channel, fringe};
use crate::dimension::{
    CAPACITANCE, CURRENT, LENGTH, MOBILITY, PER_KELVIN, PER_VOLT, RESISTANCE, TEMPERATURE, TIME,
    VOLTAGE, VOLTS_PER_KELVIN,
};
use crate::kernel::jfet::Channel;
use crate::{Lift, REFERENCE_TEMPERATURE, Real};

binned_parameters! {
    /// Parameters of a [`Bsim3`](super::Bsim3) transistor: instance geometry followed by
    /// the model card. Model-card fields preserve BSIM3v3.3 manual names, and every binned
    /// field accepts `l`, `w`, and `p` prefixes for length, width, and area terms in SI
    /// units (reciprocal meters rather than micrometers).
    #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, Lift)]
    #[serde(default, deny_unknown_fields)]
    #[schemars(bound = "R: Serialize + JsonSchema")]
    pub struct Bsim3Parameters<R: Real = f64> {
        /// The channel; default `n`.
        pub channel: Channel = Channel::N,
        /// The channel's drawn width, `w`, in meters; default 5 µm.
        pub width: R = R::constant(5e-6) [LENGTH],
        /// The channel's drawn length, `l`, in meters; default 5 µm.
        pub length: R = R::constant(5e-6) [LENGTH],
        /// The number of identical transistors in parallel, `m`; default 1.
        pub multiplier: R = R::constant(1.0),
        /// The drain diffusion's area, `ad`, in square meters; default 0.
        pub drain_area: R = R::constant(0.0) [LENGTH.pow(2)],
        /// The source diffusion's area, `as`, in square meters; default 0.
        pub source_area: R = R::constant(0.0) [LENGTH.pow(2)],
        /// The drain diffusion's perimeter, `pd`, in meters; default 0.
        pub drain_perimeter: R = R::constant(0.0) [LENGTH],
        /// The source diffusion's perimeter, `ps`, in meters; default 0.
        pub source_perimeter: R = R::constant(0.0) [LENGTH],
        /// The drain diffusion's length in squares, `nrd`; default 1.
        pub drain_squares: R = R::constant(1.0),
        /// The source diffusion's length in squares, `nrs`; default 1.
        pub source_squares: R = R::constant(1.0),
        /// A shift of the zero-bias threshold, `delvto`, in volts; default 0.
        pub threshold_adjustment: R = R::constant(0.0) [VOLTAGE],
        /// A factor of the low-field mobility, `mulu0`; default 1.
        pub mobility_multiplier: R = R::constant(1.0),
        /// The mobility model, `mobmod`, 1, 2 or 3; default 1.
        pub mobility_model: Mobility = Mobility::One,
        /// The charge model, `capmod`, 0 to 3; default 3.
        pub capacitance_model: Charges = Charges::Three,
        /// The noise model, `noimod`, 1 to 6; default 1.
        pub noise_model: NoiseModel = NoiseModel::One,
        /// Flag enabling range checking and clamping of size-dependent parameters
        /// (`paramchk = 1`); default false.
        pub check_parameters: bool = false,
        /// The share of the channel's charge the drain takes in saturation, `xpart`: 0.4 below
        /// 0.5, 0.5 at 0.5 and none above it, and no channel charge at all below 0; default 0.
        pub charge_partition: f64 = 0.0,
        /// The temperature the parameters were measured at, `tnom`, in kelvins; default 27 °C.
        pub nominal_temperature: R = R::constant(REFERENCE_TEMPERATURE) [TEMPERATURE],
        /// The gate oxide's thickness, `tox`, in meters; default 1.5e-8.
        pub tox: R = R::constant(1.5e-8) [LENGTH],
        /// The oxide thickness the parameters were extracted at, which scales `k1` and
        /// `k2`, `toxm`, in meters; default `tox`.
        pub toxm: Option<R> = None [LENGTH],
        /// The source and drain junctions' depth, `xj`, in meters; default 1.5e-7.
        binned pub xj: R = R::constant(1.5e-7) [LENGTH],
        /// The channel's doping concentration near the surface, `nch`, per cubic meter;
        /// default 1.7e23 per cubic meter, or derived from `gamma1` where it alone is
        /// given.
        binned pub nch: Option<R> = None => R::constant(1.7e23) [LENGTH.pow(-3)],
        /// The substrate's doping concentration, `nsub`, per cubic meter; default 6e22.
        binned pub nsub: R = R::constant(6.0e22) [LENGTH.pow(-3)],
        /// The poly gate's doping concentration, zero for no depletion of the gate,
        /// `ngate`, per cubic meter; default 0.
        binned pub ngate: R = R::constant(0.0) [LENGTH.pow(-3)],
        /// The body effect near the surface, `gamma1`, in square-root volts; default
        /// derived from `nch`.
        binned pub gamma1: Option<R> = None [VOLTAGE.sqrt()],
        /// The body effect in the bulk, `gamma2`, in square-root volts; default derived
        /// from `nsub`.
        binned pub gamma2: Option<R> = None [VOLTAGE.sqrt()],
        /// The bulk-source voltage at which the depletion reaches `xt`, `vbx`, in volts;
        /// default derived from `nch` and `xt`.
        binned pub vbx: Option<R> = None [VOLTAGE],
        /// The most negative bulk-source voltage the threshold's body effect reads, `vbm`,
        /// in volts; default -3.
        binned pub vbm: R = R::constant(-3.0) [VOLTAGE],
        /// The depth of the doping near the surface, `xt`, in meters; default 1.55e-7.
        binned pub xt: R = R::constant(1.55e-7) [LENGTH],
        /// The zero-bias threshold voltage of a long, wide channel, negative for a
        /// p-channel, `vth0`, in volts; default derived from `vfb`, `phi` and `k1`.
        binned pub vth0: Option<R> = None [VOLTAGE],
        /// The flat-band voltage, `vfb`, in volts; default derived from `vth0`, or -1 V.
        binned pub vfb: Option<R> = None [VOLTAGE],
        /// The first-order body effect, `k1`, in square-root volts; default derived from
        /// `gamma1` and `gamma2`, or 0.53 √V where only `k2` is given.
        binned pub k1: Option<R> = None [VOLTAGE.sqrt()],
        /// The second-order body effect, `k2`; default derived from `gamma1` and `gamma2`,
        /// or -0.0186 where only `k1` is given.
        binned pub k2: Option<R> = None,
        /// The narrow-width coefficient, `k3`; default 80.
        binned pub k3: R = R::constant(80.0),
        /// The body effect on `k3`, `k3b`, per volt; default 0.
        binned pub k3b: R = R::constant(0.0) [PER_VOLT],
        /// The width offset of the narrow-width effect, `w0`, in meters; default 2.5e-6.
        binned pub w0: R = R::constant(2.5e-6) [LENGTH],
        /// The lateral non-uniform doping along the channel, `nlx`, in meters; default
        /// 1.74e-7.
        binned pub nlx: R = R::constant(1.74e-7) [LENGTH],
        /// The first coefficient of the short-channel effect, `dvt0`; default 2.2.
        binned pub dvt0: R = R::constant(2.2),
        /// The second coefficient of the short-channel effect, `dvt1`; default 0.53.
        binned pub dvt1: R = R::constant(0.53),
        /// The body effect on the short-channel effect, `dvt2`, per volt; default -0.032.
        binned pub dvt2: R = R::constant(-0.032) [PER_VOLT],
        /// The first coefficient of the narrow-width effect on a short channel, `dvt0w`;
        /// default 0.
        binned pub dvt0w: R = R::constant(0.0),
        /// The second coefficient of the narrow-width effect on a short channel, `dvt1w`,
        /// per meter; default 5300000.
        binned pub dvt1w: R = R::constant(5.3e6) [LENGTH.pow(-1)],
        /// The body effect on the narrow-width effect on a short channel, `dvt2w`, per
        /// volt; default -0.032.
        binned pub dvt2w: R = R::constant(-0.032) [PER_VOLT],
        /// The drain-induced barrier lowering's coefficient below the threshold, `dsub`;
        /// default `drout`.
        binned pub dsub: Option<R> = None => self.drout,
        /// The drain-induced barrier lowering below the threshold, `eta0`; default 0.08.
        binned pub eta0: R = R::constant(0.08),
        /// The body effect on the drain-induced barrier lowering below the threshold,
        /// `etab`, per volt; default -0.07.
        binned pub etab: R = R::constant(-0.07) [PER_VOLT],
        /// The low-field mobility at the nominal temperature, `u0`, in square meters per
        /// volt-second; default 0.067 m²/(V·s) for an n-channel, 0.025 m²/(V·s) for a
        /// p-channel.
        binned pub u0: Option<R> = None
            => R::constant(by_channel(self.channel, 0.067, 0.025)) [MOBILITY],
        /// The first-order degradation of the mobility by the gate field, `ua`, in meters
        /// per volt; default 2.25e-9.
        binned pub ua: R = R::constant(2.25e-9) [LENGTH.div(VOLTAGE)],
        /// The second-order degradation of the mobility by the gate field, `ub`, in square
        /// meters per square volt; default 5.87e-19.
        binned pub ub: R = R::constant(5.87e-19) [LENGTH.div(VOLTAGE).pow(2)],
        /// The body effect on the mobility's degradation, `uc`, in meters per square volt,
        /// or per volt at `mobmod` 3; default -4.65e-11 m/V², or -0.0465 per volt at
        /// `mobmod` 3.
        binned pub uc: Option<R> = None
            => R::constant(self.mobility_model.select(-0.0465e-9, -0.0465)) [conditional("uc")],
        /// The carriers' saturation velocity at the nominal temperature, `vsat`, in meters
        /// per second; default 80000.
        binned pub vsat: R = R::constant(80000.0) [LENGTH.div(TIME)],
        /// The bulk charge effect of a short channel, `a0`; default 1.
        binned pub a0: R = R::constant(1.0),
        /// The gate bias's effect on the bulk charge, `ags`, per volt; default 0.
        binned pub ags: R = R::constant(0.0) [PER_VOLT],
        /// The first coefficient of the non-saturation effect, `a1`, per volt; default 0.
        binned pub a1: R = R::constant(0.0) [PER_VOLT],
        /// The second coefficient of the non-saturation effect, `a2`; default 1.
        binned pub a2: R = R::constant(1.0),
        /// The narrow-width effect on the bulk charge, `b0`, in meters; default 0.
        binned pub b0: R = R::constant(0.0) [LENGTH],
        /// The width offset of the narrow-width effect on the bulk charge, `b1`, in meters;
        /// default 0.
        binned pub b1: R = R::constant(0.0) [LENGTH],
        /// The body effect on the bulk charge, `keta`, per volt; default -0.047.
        binned pub keta: R = R::constant(-0.047) [PER_VOLT],
        /// Subthreshold offset voltage in drain-current smoothing, `voff`, in volts;
        /// default -0.08.
        binned pub voff: R = R::constant(-0.08) [VOLTAGE],
        /// The subthreshold swing's coefficient, `nfactor`; default 1.
        binned pub nfactor: R = R::constant(1.0),
        /// The coupling capacitance between the channel and the source and drain, `cdsc`,
        /// in farads per square meter; default 0.00024.
        binned pub cdsc: R = R::constant(0.00024) [CAPACITANCE.div(LENGTH.pow(2))],
        /// The body bias's effect on `cdsc`, `cdscb`, in farads per volt-square-meter;
        /// default 0.
        binned pub cdscb: R = R::constant(0.0) [CAPACITANCE.div(VOLTAGE.mul(LENGTH.pow(2)))],
        /// The drain bias's effect on `cdsc`, `cdscd`, in farads per volt-square-meter;
        /// default 0.
        binned pub cdscd: R = R::constant(0.0) [CAPACITANCE.div(VOLTAGE.mul(LENGTH.pow(2)))],
        /// The interface traps' capacitance, `cit`, in farads per square meter; default 0.
        binned pub cit: R = R::constant(0.0) [CAPACITANCE.div(LENGTH.pow(2))],
        /// The channel-length modulation's coefficient, `pclm`; default 1.3.
        binned pub pclm: R = R::constant(1.3),
        /// The first coefficient of the drain-induced barrier lowering on the output
        /// resistance, `pdiblc1`; default 0.39.
        binned pub pdiblc1: R = R::constant(0.39),
        /// The second coefficient of the drain-induced barrier lowering on the output
        /// resistance, `pdiblc2`; default 0.0086.
        binned pub pdiblc2: R = R::constant(0.0086),
        /// The body effect on the drain-induced barrier lowering on the output resistance,
        /// `pdiblcb`, per volt; default 0.
        binned pub pdiblcb: R = R::constant(0.0) [PER_VOLT],
        /// The length dependence of the drain-induced barrier lowering on the output
        /// resistance, `drout`; default 0.56.
        binned pub drout: R = R::constant(0.56),
        /// The first coefficient of the substrate current's body effect on the output
        /// resistance, `pscbe1`, in volts per meter; default 424000000.
        binned pub pscbe1: R = R::constant(4.24e8) [VOLTAGE.div(LENGTH)],
        /// The second coefficient of the substrate current's body effect on the output
        /// resistance, `pscbe2`, in meters per volt; default 1e-5.
        binned pub pscbe2: R = R::constant(1.0e-5) [LENGTH.div(VOLTAGE)],
        /// The gate bias's effect on the Early voltage, `pvag`; default 0.
        binned pub pvag: R = R::constant(0.0),
        /// The smoothing of the effective drain-source voltage at saturation, `delta`, in
        /// volts; default 0.01.
        binned pub delta: R = R::constant(0.01) [VOLTAGE],
        /// The source and drain resistance of a channel 1 µm wide, which the drain current
        /// folds in, `RDSW` in ohms times micrometers to the power `wr`, `rdsw`, in ohms;
        /// default 0.
        binned pub rdsw: R = R::constant(0.0) [RESISTANCE],
        /// The gate bias's effect on `rdsw`, `prwg`, per volt; default 0.
        binned pub prwg: R = R::constant(0.0) [PER_VOLT],
        /// The body effect on `rdsw`, `prwb`, per square-root volt; default 0.
        binned pub prwb: R = R::constant(0.0) [VOLTAGE.sqrt().recip()],
        /// The width exponent of `rdsw`, `wr`; default 1.
        binned pub wr: R = R::constant(1.0),
        /// The gate bias's narrowing of the channel, `dwg`, in meters per volt; default 0.
        binned pub dwg: R = R::constant(0.0) [LENGTH.div(VOLTAGE)],
        /// The body bias's narrowing of the channel, `dwb`, in meters per square-root volt;
        /// default 0.
        binned pub dwb: R = R::constant(0.0) [LENGTH.div(VOLTAGE.sqrt())],
        /// The first coefficient of the impact ionization's substrate current, `alpha0`, in
        /// meters per volt; default 0.
        binned pub alpha0: R = R::constant(0.0) [LENGTH.div(VOLTAGE)],
        /// The length scaling of the impact ionization's substrate current, `alpha1`, per
        /// volt; default 0.
        binned pub alpha1: R = R::constant(0.0) [PER_VOLT],
        /// The second coefficient of the impact ionization's substrate current, `beta0`, in
        /// volts; default 30.
        binned pub beta0: R = R::constant(30.0) [VOLTAGE],
        /// The threshold's temperature coefficient, `kt1`, in volts; default -0.11.
        binned pub kt1: R = R::constant(-0.11) [VOLTAGE],
        /// The channel length's effect on `kt1`, `kt1l`, in volt-meters; default 0.
        binned pub kt1l: R = R::constant(0.0) [VOLTAGE.mul(LENGTH)],
        /// The body effect on `kt1`, `kt2`; default 0.022.
        binned pub kt2: R = R::constant(0.022),
        /// The mobility's temperature exponent, `ute`; default -1.5.
        binned pub ute: R = R::constant(-1.5),
        /// The temperature coefficient of `ua`, `ua1`, in meters per volt; default 4.31e-9.
        binned pub ua1: R = R::constant(4.31e-9) [LENGTH.div(VOLTAGE)],
        /// The temperature coefficient of `ub`, `ub1`, in square meters per square volt;
        /// default -7.61e-18.
        binned pub ub1: R = R::constant(-7.61e-18) [LENGTH.div(VOLTAGE).pow(2)],
        /// The temperature coefficient of `uc`, `uc1`, in meters per square volt, or per
        /// volt at `mobmod` 3; default -5.6e-11 m/V², or -0.056 per volt at `mobmod` 3.
        binned pub uc1: Option<R> = None
            => R::constant(self.mobility_model.select(-0.056e-9, -0.056)) [conditional("uc")],
        /// The temperature coefficient of `vsat`, `at`, in meters per second; default
        /// 33000.
        binned pub at: R = R::constant(33000.0) [LENGTH.div(TIME)],
        /// Temperature coefficient of `rdsw`, `prt`, in ohms; default 0.
        binned pub prt: R = R::constant(0.0) [RESISTANCE],
        /// The bias-dependent overlap capacitance between the gate and the lightly doped
        /// source, `cgsl`, in farads per meter; default 0.
        binned pub cgsl: R = R::constant(0.0) [CAPACITANCE.div(LENGTH)],
        /// The bias-dependent overlap capacitance between the gate and the lightly doped
        /// drain, `cgdl`, in farads per meter; default 0.
        binned pub cgdl: R = R::constant(0.0) [CAPACITANCE.div(LENGTH)],
        /// The coefficient of the bias-dependent overlap capacitances, `ckappa`, in volts;
        /// default 0.6.
        binned pub ckappa: R = R::constant(0.6) [VOLTAGE],
        /// The fringing capacitance per width, `cf`, in farads per meter; default 2 · εox /
        /// π · ln(1 + 0.4 µm / `tox`).
        binned pub cf: Option<R> = None => fringe(self.tox) [CAPACITANCE.div(LENGTH)],
        /// The first coefficient of the short-channel effect on the charges, `clc`, in
        /// meters; default 1e-7.
        binned pub clc: R = R::constant(1.0e-7) [LENGTH],
        /// The second coefficient of the short-channel effect on the charges, `cle`;
        /// default 0.6.
        binned pub cle: R = R::constant(0.6),
        /// The flat-band voltage of `capmod` 0, `vfbcv`, in volts; default -1.
        binned pub vfbcv: R = R::constant(-1.0) [VOLTAGE],
        /// Charge-layer thickness exponential coefficient for `capmod` 3, `acde`, in meters
        /// per volt; default 1.
        binned pub acde: R = R::constant(1.0) [LENGTH.div(VOLTAGE)],
        /// Surface-potential gate-bias coefficient for `capmod` 3, `moin`; default 15.
        binned pub moin: R = R::constant(15.0),
        /// Subthreshold swing coefficient in `capmod` 1 to 3 charges, `noff`; default 1.
        binned pub noff: R = R::constant(1.0),
        /// Threshold offset voltage in `capmod` 1 to 3 charges, `voffcv`, in volts;
        /// default 0.
        binned pub voffcv: R = R::constant(0.0) [VOLTAGE],
        /// Fixed gate-source overlap capacitance per unit width, `cgso`, in farads per
        /// meter; default `dlc` · Cox − `cgsl` when `dlc > 0`, otherwise 0.6 · `xj` · Cox.
        pub cgso: Option<R> = None [CAPACITANCE.div(LENGTH)],
        /// Fixed gate-drain overlap capacitance per unit width, `cgdo`, in farads per
        /// meter; default `dlc` · Cox − `cgdl` when `dlc > 0`, otherwise 0.6 · `xj` · Cox.
        pub cgdo: Option<R> = None [CAPACITANCE.div(LENGTH)],
        /// The gate-bulk overlap capacitance per length, `cgbo`, in farads per meter;
        /// default 2 · `dwc` · Cox.
        pub cgbo: Option<R> = None [CAPACITANCE.div(LENGTH)],
        /// The length each side of the channel loses, which shortens it, `lint`, in meters;
        /// default 0.
        pub lint: R = R::constant(0.0) [LENGTH],
        /// The length dependence of the length lost, `ll`, in meters to the power 1 +
        /// `lln`; default 0.
        pub ll: R = R::constant(0.0),
        /// The length exponent of `ll`, `lln`; default 1.
        pub lln: R = R::constant(1.0),
        /// The width dependence of the length lost, `lw`, in meters times meters to the
        /// power `lwn`; default 0.
        pub lw: R = R::constant(0.0),
        /// The width exponent of `lw`, `lwn`; default 1.
        pub lwn: R = R::constant(1.0),
        /// The area dependence of the length lost, `lwl`, in meters times meters to the
        /// power `lln` + `lwn`; default 0.
        pub lwl: R = R::constant(0.0),
        /// The length dependence of the length the charges lose, `llc`, in meters to the
        /// power 1 + `lln`; default `ll`.
        pub llc: Option<R> = None,
        /// The width dependence of the length the charges lose, `lwc`, in meters times
        /// meters to the power `lwn`; default `lw`.
        pub lwc: Option<R> = None,
        /// The area dependence of the length the charges lose, `lwlc`, in meters times
        /// meters to the power `lln` + `lwn`; default `lwl`.
        pub lwlc: Option<R> = None,
        /// The width each side of the channel loses, which narrows it, `wint`, in meters;
        /// default 0.
        pub wint: R = R::constant(0.0) [LENGTH],
        /// The length dependence of the width lost, `wl`, in meters times meters to the
        /// power `wln`; default 0.
        pub wl: R = R::constant(0.0),
        /// The length exponent of `wl`, `wln`; default 1.
        pub wln: R = R::constant(1.0),
        /// The width dependence of the width lost, `ww`, in meters to the power 1 + `wwn`;
        /// default 0.
        pub ww: R = R::constant(0.0),
        /// The width exponent of `ww`, `wwn`; default 1.
        pub wwn: R = R::constant(1.0),
        /// The area dependence of the width lost, `wwl`, in meters times meters to the
        /// power `wln` + `wwn`; default 0.
        pub wwl: R = R::constant(0.0),
        /// The length dependence of the width the charges lose, `wlc`, in meters times
        /// meters to the power `wln`; default `wl`.
        pub wlc: Option<R> = None,
        /// The width dependence of the width the charges lose, `wwc`, in meters to the
        /// power 1 + `wwn`; default `ww`.
        pub wwc: Option<R> = None,
        /// The area dependence of the width the charges lose, `wwlc`, in meters times
        /// meters to the power `wln` + `wwn`; default `wwl`.
        pub wwlc: Option<R> = None,
        /// The length each side of the channel loses in the charges, `dlc`, in meters;
        /// default `lint`.
        pub dlc: Option<R> = None [LENGTH],
        /// The width each side of the channel loses in the charges, `dwc`, in meters;
        /// default `wint`.
        pub dwc: Option<R> = None [LENGTH],
        /// The length the mask and etching add, `xl`, in meters; default 0.
        pub xl: R = R::constant(0.0) [LENGTH],
        /// The width the mask and etching add, `xw`, in meters; default 0.
        pub xw: R = R::constant(0.0) [LENGTH],
        /// The source and drain diffusions' sheet resistance, `rsh`, in ohms per square;
        /// default 0.
        pub rsh: R = R::constant(0.0) [RESISTANCE],
        /// The junctions' saturation current per area, `js`, in amperes per square meter;
        /// default 0.0001.
        pub js: R = R::constant(0.0001) [CURRENT.div(LENGTH.pow(2))],
        /// The junctions' saturation current per length of perimeter, `jsw`, in amperes per
        /// meter; default 0.
        pub jsw: R = R::constant(0.0) [CURRENT.div(LENGTH)],
        /// The junction current above which the diodes' exponential continues linearly,
        /// zero for none, `ijth`, in amperes; default 0.1.
        pub ijth: R = R::constant(0.1) [CURRENT],
        /// The junctions' emission coefficient, `nj`; default 1.
        pub nj: R = R::constant(1.0),
        /// The junctions' saturation current temperature exponent, `xti`; default 3.
        pub xti: R = R::constant(3.0),
        /// The junctions' zero-bias bottom capacitance per area, `cj`, in farads per square
        /// meter; default 0.0005.
        pub cj: R = R::constant(0.0005) [CAPACITANCE.div(LENGTH.pow(2))],
        /// The junctions' zero-bias sidewall capacitance per length along the field oxide,
        /// `cjsw`, in farads per meter; default 5e-10.
        pub cjsw: R = R::constant(5.0e-10) [CAPACITANCE.div(LENGTH)],
        /// The junctions' zero-bias sidewall capacitance per length along the gate,
        /// `cjswg`, in farads per meter; default `cjsw`.
        pub cjswg: Option<R> = None [CAPACITANCE.div(LENGTH)],
        /// The bottom capacitance's grading coefficient, `mj`; default 0.5.
        pub mj: R = R::constant(0.5),
        /// The field-oxide sidewall's grading coefficient, `mjsw`; default 0.33.
        pub mjsw: R = R::constant(0.33),
        /// The gate sidewall's grading coefficient, `mjswg`; default `mjsw`.
        pub mjswg: Option<R> = None,
        /// The bottom junction's built-in potential, at least 0.1 V, `pb`, in volts;
        /// default 1.
        pub pb: R = R::constant(1.0) [VOLTAGE],
        /// The field-oxide sidewall's built-in potential, at least 0.1 V, `pbsw`, in volts;
        /// default 1.
        pub pbsw: R = R::constant(1.0) [VOLTAGE],
        /// The gate sidewall's built-in potential, at least 0.1 V, `pbswg`, in volts;
        /// default `pbsw`.
        pub pbswg: Option<R> = None [VOLTAGE],
        /// The temperature coefficient of `pb`, `tpb`, in volts per kelvin; default 0.
        pub tpb: R = R::constant(0.0) [VOLTS_PER_KELVIN],
        /// The temperature coefficient of `pbsw`, `tpbsw`, in volts per kelvin; default 0.
        pub tpbsw: R = R::constant(0.0) [VOLTS_PER_KELVIN],
        /// The temperature coefficient of `pbswg`, `tpbswg`, in volts per kelvin; default
        /// 0.
        pub tpbswg: R = R::constant(0.0) [VOLTS_PER_KELVIN],
        /// The temperature coefficient of `cj`, `tcj`, per kelvin; default 0.
        pub tcj: R = R::constant(0.0) [PER_KELVIN],
        /// The temperature coefficient of `cjsw`, `tcjsw`, per kelvin; default 0.
        pub tcjsw: R = R::constant(0.0) [PER_KELVIN],
        /// The temperature coefficient of `cjswg`, `tcjswg`, per kelvin; default 0.
        pub tcjswg: R = R::constant(0.0) [PER_KELVIN],
        /// First oxide-trap density coefficient for flicker noise (`noimod` 2, 3, and 6),
        /// `noia`, in manual units; default 1e20 for n-channel, 9.9e18 for p-channel.
        pub noia: Option<R> = None,
        /// Second oxide-trap density coefficient for flicker noise, `noib`, in manual
        /// units; default 5e4 for n-channel, 2.4e3 for p-channel.
        pub noib: Option<R> = None,
        /// Third oxide-trap density coefficient for flicker noise, `noic`, in manual units;
        /// default -1.4e-12 for n-channel, 1.4e-12 for p-channel.
        pub noic: Option<R> = None,
        /// Saturation field for flicker noise, `em`, in volts per meter; default 41000000.
        pub em: R = R::constant(4.1e7) [VOLTAGE.div(LENGTH)],
        /// Flicker-noise frequency exponent, `ef`; default 1.
        pub ef: R = R::constant(1.0),
        /// Flicker-noise current exponent (`noimod` 1, 4, and 5), `af`; default 1.
        pub af: R = R::constant(1.0),
        /// Flicker-noise coefficient (`noimod` 1, 4, and 5), `kf`; default 0.
        pub kf: R = R::constant(0.0),
        /// Channel length offset per side for flicker noise beyond `lint`, `lintnoi`, in
        /// meters; default 0.
        pub lintnoi: R = R::constant(0.0) [LENGTH],
    }

    /// Binned parameters evaluated at a specific channel size with length, width, and area
    /// terms summed. Parameters with derived defaults remain `None` when omitted from the
    /// model card so ngspice derivation rules take precedence over binning terms.
    #[derive(Debug, Clone, Copy)]
    pub(super) struct Binned<R: Real>;
}
