//! A BSIM4 transistor's numbers at its temperature and size, as ngspice 47's
//! `b4v5temp.c` prepares them, with `b4v5geo.c`'s diffusions, checked as its
//! `b4v5check.c` checks them.
//!
//! Part of jolty's BSIM4, derived from BSIM4.5.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2005 Regents of the University of California. All rights reserved. Used
//! under the BSIM4 terms of use, which the repository's `NOTICE` reproduces.

use super::channel::theta;
use super::charges::Depletion;
use super::parameters::{Binned, Bsim4Parameters};
use super::{
    BodyResistance, DrainSourceResistance, EPS0, FlickerNoise, PI, Perimeter, ThermalNoise,
};
use super::{D, dexp};
use crate::kernel::bsim3::{EPSSI, KBOQ, Q, by_channel, check, lost};
use crate::kernel::parts::series_value;
use crate::{ParameterError, Real};

/// The intrinsic carrier concentration of silicon at 300.15 K, per cubic centimeter, as
/// `b4v5temp.c` takes it.
const INTRINSIC: f64 = 1.45e10;
/// The smoothing `b4v5temp.c` adds to the stress effect's factors, its `DELTA`.
const STRESS_DELTA: f64 = 1.0e-9;
/// The oxide thickness `b4v5set.c` defaults `toxe` to, in meters.
const TOXE: f64 = 30.0e-10;

/// The numbers the drain current reads, its channel's size and temperature folded in.
/// Concentrations are per cubic centimeter, as the equations of `b4v5ld.c` take them.
#[derive(Debug, Clone, Copy)]
pub(super) struct Core<R: Real> {
    /// The channel's sign, 1 for an n-channel and −1 for a p-channel.
    pub(super) sign: f64,
    pub(super) toxe: R,
    pub(super) toxp: R,
    pub(super) coxe: R,
    pub(super) coxp: R,
    /// `√(εsi / (epsrox · ε0) · toxe)`.
    pub(super) factor1: R,
    /// The thermal voltage `k·T / q` at the temperature, with BSIM4's `k / q`.
    pub(super) vtm: R,
    /// `T / Tnom − 1`.
    pub(super) temperature_rise: R,
    pub(super) leff: R,
    pub(super) weff: R,
    pub(super) weff_cj: R,
    pub(super) leff_cv: R,
    pub(super) weff_cv: R,
    pub(super) fingers: R,
    /// The drain's share of the channel's charge in saturation, `xpart`.
    pub(super) xpart: f64,
    pub(super) abulk_cv_factor: R,
    /// The flat-band voltage at zero bias of the charges.
    pub(super) vfbzb: R,
    /// The Debye length of the charge-thickness model, and its exponential coefficient
    /// `acde` scaled by the doping.
    pub(super) ldeb: R,
    pub(super) acde: R,
    pub(super) phi: R,
    pub(super) sqrt_phi: R,
    pub(super) xdep0: R,
    pub(super) vbi: R,
    pub(super) cdep0: R,
    pub(super) litl: R,
    pub(super) k1: R,
    pub(super) k1ox: R,
    pub(super) k2ox: R,
    pub(super) mstar: R,
    pub(super) voffcbn: R,
    pub(super) theta0vb0: R,
    pub(super) theta_rout: R,
    /// `rdsw`'s resistance within the drain current, none at `rdsmod` 1.
    pub(super) rds0: R,
    pub(super) rdswmin: R,
    pub(super) ua: R,
    pub(super) ub: R,
    pub(super) uc: R,
    pub(super) ud: R,
    pub(super) u0temp: R,
    pub(super) vsattemp: R,
    /// `(1 − T0) / (1 + T0)` of the source end's velocity limit, where a positive `vtl`
    /// is given.
    pub(super) tfactor: Option<R>,
    /// The threshold at zero bias of a long, wide channel, negative for a p-channel as the
    /// card's `vth0` is, the stress effect and `delvto` added.
    pub(super) vth0: R,
    pub(super) vfb: R,
    pub(super) eta0: R,
    pub(super) vbsc: R,
    pub(super) vtfbphi2: R,
    /// The poly gate's doping, per cubic centimeter.
    pub(super) ngate: R,
    pub(super) pditsl: R,
    /// The velocity overshoot's binned `lambda`, where a positive one is given.
    pub(super) lambda: Option<R>,
    pub(super) binned: Binned<R>,
}

/// One junction between a body node and the drain or the source.
#[derive(Debug, Clone, Copy)]
pub(super) struct Junction<R: Real> {
    pub(super) saturation_current: R,
    /// `Nvtm`, the emission coefficient times the thermal voltage.
    pub(super) slope: R,
    /// The voltage above which the current continues linearly, and the current
    /// `Is · e^(vjm / Nvtm)` there.
    pub(super) vjm: R,
    pub(super) current_at_vjm: R,
    /// The trap-assisted tunneling of the bottom, the isolation sidewall and the gate
    /// sidewall: each one's saturation current, its `vts` and its slope `Nvtm0 · njts`.
    pub(super) tunneling: [(R, R, R); 3],
}

/// The overlaps between the gate and the source, the drain and the body.
#[derive(Debug, Clone, Copy)]
pub(super) struct Overlaps<R: Real> {
    /// `(cgso + cf) · WeffCV` and `(cgdo + cf) · WeffCV`.
    pub(super) cgso: R,
    pub(super) cgdo: R,
    /// `cgsl · WeffCV` and `cgdl · WeffCV`.
    pub(super) cgsl: R,
    pub(super) cgdl: R,
    pub(super) ckappas: R,
    pub(super) ckappad: R,
    /// `cgbo · LeffCV · nf`.
    pub(super) cgbo: R,
    pub(super) fingers: R,
}

/// The bias-dependent drain and source resistances of `rdsmod` 1: of each one copy at
/// zero and at high gate bias, `rd0` and `rdwmin` and `rs0` and `rswmin`, the flat-band
/// voltage of the diffusions `vfbsd`, the bias coefficients `prwg` and `prwb`, and the
/// conductances of `rsh` behind the drain and the source, 1 kS where `nrd` or `nrs` sets
/// none, as `b4v5temp.c` resets them.
#[derive(Debug, Clone, Copy)]
pub(super) struct External<R: Real> {
    pub(super) minimums: [R; 2],
    pub(super) resistances: [R; 2],
    pub(super) vfbsd: R,
    pub(super) prwg: R,
    pub(super) prwb: R,
    pub(super) sheet: [R; 2],
}

/// The numbers of the noise models.
#[derive(Debug, Clone, Copy)]
pub(super) struct Noise {
    pub(super) flicker_model: FlickerNoise,
    pub(super) thermal_model: ThermalNoise,
    pub(super) noia: f64,
    pub(super) noib: f64,
    pub(super) noic: f64,
    pub(super) em: f64,
    pub(super) ef: f64,
    pub(super) af: f64,
    pub(super) kf: f64,
    pub(super) ntnoi: f64,
    pub(super) tnoia: f64,
    pub(super) tnoib: f64,
    pub(super) rnoia: f64,
    pub(super) rnoib: f64,
    /// The effective length less twice `lintnoi`.
    pub(super) length: f64,
}

/// A transistor's numbers at its temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedBsim4<R: Real> {
    pub(super) core: Core<R>,
    pub(super) overlaps: Overlaps<R>,
    pub(super) drain_depletion: Depletion<R>,
    pub(super) source_depletion: Depletion<R>,
    pub(super) drain_junction: Junction<R>,
    pub(super) source_junction: Junction<R>,
    /// The drain and source resistances' stamped values: each conductance, or each
    /// resistance where its current is an unknown, of the parallel copies together.
    pub(super) series: [R; 2],
    pub(super) external: Option<External<R>>,
    /// The drain and source resistances, of the parallel copies together, none where
    /// there is no node behind them.
    pub(super) resistances: [R; 2],
    pub(super) noise: Noise,
    pub(super) temperature: f64,
    /// The conductances of the five body resistances of `rbodymod` 1, in the order of
    /// `BODY_RESISTANCES`, of the parallel copies together.
    pub(super) body_network: Option<[R; 5]>,
    pub(super) multiplier: R,
}

/// The arctangent of `x`.
fn atan<R: Real>(x: R) -> R {
    let value = x.value();
    let slope = 1.0 / (1.0 + value * value);
    x.chain(value.atan(), slope, -2.0 * value * slope * slope)
}

/// The source's and the drain's diffusions of `fingers` fingers, `b4v5geo.c`'s
/// `BSIM4v5PAeffGeo` at `geomod` 0 with `min` 0: their perimeters and areas, from the
/// junctions' width `weff_cj` and the distances `dmcg`, less `dmcgt`, and `dmci`.
#[expect(clippy::similar_names, reason = "b4v5geo.c's names of the diffusions")]
fn diffusions<R: Real>(fingers: f64, weff_cj: R, [dmcg, dmci]: [R; 2]) -> [R; 4] {
    // `BSIM4v5NumFingerDiff`: an odd number of fingers ends in a source and a drain, an
    // even one in two drains with `min` 0.
    let whole = fingers.trunc();
    let (end_drain, inner_drain, end_source, inner_source) = if whole % 2.0 == 0.0 {
        (0.0, fingers, 2.0, 2.0 * (fingers / 2.0 - 1.0).max(0.0))
    } else {
        let inner = 2.0 * ((fingers - 1.0) / 2.0).max(0.0);
        (1.0, inner, 1.0, inner)
    };
    let t0 = dmcg + dmci;
    let isolated_perimeter = t0 + t0 + weff_cj;
    let shared_perimeter = dmcg + dmcg;
    let isolated_area = t0 * weff_cj;
    let shared_area = dmcg * weff_cj;
    [
        isolated_perimeter * end_source + shared_perimeter * inner_source,
        isolated_perimeter * end_drain + shared_perimeter * inner_drain,
        isolated_area * end_source + shared_area * inner_source,
        isolated_area * end_drain + shared_area * inner_drain,
    ]
}

impl<R: Real> Bsim4Parameters<R> {
    /// The thicknesses of the oxide, `toxe` and `toxp`, as `b4v5temp.c` resolves them
    /// from each other and `dtox`.
    #[expect(
        clippy::similar_names,
        reason = "the oxide thicknesses' names in BSIM4"
    )]
    fn thicknesses(&self) -> [R; 2] {
        match (self.toxe, self.toxp) {
            (Some(toxe), None) => [toxe, toxe - self.dtox],
            (None, Some(toxp)) => [toxp + self.dtox, toxp],
            (Some(toxe), Some(toxp)) => [toxe, toxp],
            (None, None) => [R::constant(TOXE); 2],
        }
    }

    /// The `toxe` that `b4v5set.c` defaults others from, before `b4v5temp.c` resolves it
    /// against `toxp`: the card's, or 3 nm.
    pub(super) fn set_toxe(&self) -> R {
        self.toxe.unwrap_or(R::constant(TOXE))
    }

    /// The drain and source resistances of `rsh` behind `nrd` and `nrs`, where each is
    /// positive, as `b4v5set.c` makes nodes for them at `rdsmod` 0 and `rgeomod` 0, of the
    /// parallel copies together.
    pub(super) fn resistances(&self) -> Result<[R; 2], ParameterError> {
        let sheet = self.rsh;
        let multiplier = ParameterError::positive("multiplier", self.multiplier)?;
        Ok(
            [self.drain_squares, self.source_squares].map(|squares| match squares {
                Some(squares) if sheet.value() > 0.0 && squares.value() > 0.0 => {
                    sheet * squares / multiplier
                }
                _ => R::constant(0.0),
            }),
        )
    }

    /// The transistor's numbers at `temperature`, in kelvins.
    #[expect(
        clippy::too_many_lines,
        clippy::similar_names,
        reason = "one pass through b4v5temp.c, in its order, by its names"
    )]
    pub(super) fn prepared(&self, temperature: R) -> Result<PreparedBsim4<R>, ParameterError> {
        if temperature.value() <= 0.0 {
            return Err(ParameterError(format!(
                "a BSIM4 transistor's temperature lies above 0 K, not at {:?} K",
                temperature.value()
            )));
        }
        let tnom = ParameterError::positive("nominal_temperature", self.nominal_temperature)?;
        let multiplier = ParameterError::positive("multiplier", self.multiplier)?;
        let fingers = self.fingers;
        check(fingers.value() < 1.0, || {
            format!("nf must be at least 1, not {:?}", fingers.value())
        })?;
        let sign = self.channel.sign();
        let [toxe, toxp] = self.thicknesses();
        let toxm = self.toxm.unwrap_or(self.set_toxe());
        for (name, value) in [
            ("toxe", toxe),
            ("toxp", toxp),
            ("toxm", toxm),
            ("toxref", self.toxref),
        ] {
            check(value.value() <= 0.0, || {
                format!("{name} must be positive, not {:?}", value.value())
            })?;
        }
        let coxe = self.epsrox * EPS0 / toxe;
        let coxp = self.epsrox * EPS0 / toxp;
        let ratio = temperature / tnom;
        let rise = ratio - 1.0;
        let delta_temperature = temperature - tnom;
        let factor1 = (toxe * EPSSI / (self.epsrox * EPS0)).sqrt();
        let vtm0 = tnom * KBOQ;
        let eg0 = -(tnom * tnom * 7.02e-4 / (tnom + 1108.0)) + 1.16;
        let reference = tnom / 300.15;
        let ni =
            reference * reference.sqrt() * INTRINSIC * (-(eg0 / (vtm0 * 2.0)) + 21.556_598_1).exp();
        let vtm = temperature * KBOQ;
        let eg = -(temperature * temperature * 7.02e-4 / (temperature + 1108.0)) + 1.16;

        // The diffusions' saturation current densities at the temperature, the drain's
        // defaulting to the source's.
        let source_densities = [self.jss, self.jsws, self.jswgs];
        let drain_densities = [
            self.jsd.unwrap_or(self.jss),
            self.jswd.unwrap_or(self.jsws),
            self.jswgd.unwrap_or(self.jswgs),
        ];
        let source_emission = self.njs;
        let drain_emission = self.njd.unwrap_or(self.njs);
        let source_exponent = self.xtis;
        let drain_exponent = self.xtid.unwrap_or(self.xtis);
        #[expect(
            clippy::float_cmp,
            reason = "b4v5temp.c scales the densities wherever the two differ at all"
        )]
        let at_nominal = temperature.value() == tnom.value();
        let heated = |densities: [R; 3], exponent: R, emission: R| {
            let scaled = if at_nominal {
                densities
            } else {
                let t0 = eg0 / vtm0 - eg / vtm;
                let t3 = ((t0 + exponent * ratio.ln()) / emission).exp();
                densities.map(|density| density * t3)
            };
            scaled.map(|density| {
                if density.value() < 0.0 {
                    R::constant(0.0)
                } else {
                    density
                }
            })
        };
        let source_densities = heated(source_densities, source_exponent, source_emission);
        let drain_densities = heated(drain_densities, drain_exponent, drain_emission);
        let at_least = |value: R| {
            if value.value() <= 0.0 {
                R::constant(0.1)
            } else {
                value
            }
        };
        let ijthsfwd = at_least(self.ijthsfwd);
        let ijthdfwd = at_least(self.ijthdfwd.unwrap_or(self.ijthsfwd));

        // The effective sizes.
        let length = ParameterError::positive("length", self.length)?;
        let width = ParameterError::positive("width", self.width)?;
        let l_new = length + self.xl;
        let w_new = width / fingers + self.xw;
        let [by_length, by_width] = [l_new.pow(self.lln), w_new.pow(self.lwn)];
        let [of_length, of_width] = [l_new.pow(self.wln), w_new.pow(self.wwn)];
        let dl = lost(
            self.lint,
            [self.ll, self.lw, self.lwl],
            [by_length, by_width],
        );
        let dw = lost(
            self.wint,
            [self.wl, self.ww, self.wwl],
            [of_length, of_width],
        );
        let dwc_model = self.dwc.unwrap_or(self.wint);
        let width_lost_cv = lost(
            R::constant(0.0),
            [
                self.wlc.unwrap_or(self.wl),
                self.wwc.unwrap_or(self.ww),
                self.wwlc.unwrap_or(self.wwl),
            ],
            [of_length, of_width],
        );
        let dwj = self.dwj.unwrap_or(dwc_model) + width_lost_cv;
        let dlc = lost(
            self.dlc.unwrap_or(self.lint),
            [
                self.llc.unwrap_or(self.ll),
                self.lwc.unwrap_or(self.lw),
                self.lwlc.unwrap_or(self.lwl),
            ],
            [by_length, by_width],
        );
        let leff_cv = l_new - dlc * 2.0;
        let weff_cv = w_new - (dwc_model + width_lost_cv) * 2.0;
        let leff = l_new - dl * 2.0;
        let weff = w_new - dw * 2.0;
        let weff_cj = w_new - dwj * 2.0;
        for (name, value) in [
            ("length", leff),
            ("width", weff),
            ("length of the charges", leff_cv),
            ("width of the charges", weff_cv),
            ("width of the junctions", weff_cj),
        ] {
            check(value.value() <= 0.0, || {
                format!(
                    "the effective {name} must be positive, not {:?}",
                    value.value()
                )
            })?;
        }
        let inverse = [leff.recip(), weff.recip(), (leff * weff).recip()];
        let mut binned = self.binned(inverse);

        // tempmod 0.
        let pow_weff_wr = (weff_cj * 1e6).pow(binned.wr) * fingers;
        let ua = binned.ua + binned.ua1 * rise;
        let ub = binned.ub + binned.ub1 * rise;
        let uc = binned.uc + binned.uc1 * rise;
        let ud = binned.ud + binned.ud1 * rise;
        let vsattemp = binned.vsat - binned.at * rise;
        let t10 = binned.prt * rise;
        let mut rds0 = (binned.rdsw + t10) * fingers / pow_weff_wr;
        let mut rdswmin = (self.rdswmin + t10) * fingers / pow_weff_wr;
        let t5 = -(binned.up * (-(leff / binned.lp)).exp()) + 1.0;
        let u0temp = binned.u0 * t5 * ratio.pow(binned.ute);
        binned.voff = binned.voff * (binned.tvoff * delta_temperature + 1.0);
        let limited = self.vtl.is_some_and(|vtl| vtl.value() > 0.0);
        let tfactor = limited.then(|| {
            let lc = if self.lc.value() < 0.0 {
                R::constant(0.0)
            } else {
                self.lc
            };
            let t0 = leff / (binned.xn * leff + lc);
            (-t0 + 1.0) / (t0 + 1.0)
        });

        // Concentrations per cubic centimeter, as `b4v5temp.c` and `b4v5ld.c` read them.
        let mut ndep = binned.ndep * 1e-6;
        if self.ndep.is_none()
            && let Some(gamma1) = binned.gamma1
        {
            let product = gamma1 * coxe;
            ndep = product * product * 3.012_48e22;
        }
        let nsub = binned.nsub * 1e-6;
        let nsd = binned.nsd * 1e-6;
        let ngate = binned.ngate * 1e-6;
        let phi = vtm0 * (ndep / ni).ln() + binned.phin + 0.4;
        let sqrt_phi = phi.sqrt();
        let xdep0 = (R::constant(2.0 * EPSSI / (Q * 1e6)) / ndep).sqrt() * sqrt_phi;
        let litl = (binned.xj * toxe * 3.0).sqrt();
        let vbi = vtm0 * (nsd * ndep / (ni * ni)).ln();
        let cdep0 = (ndep * (Q * EPSSI * 1e6 / 2.0) / phi).sqrt();
        let mstar = atan(binned.minv) / PI + 0.5;
        let voffcbn = binned.voff + self.voffl / leff;

        let (k1, mut k2) = if self.k1.is_some() || self.k2.is_some() {
            (
                binned.k1.unwrap_or(R::constant(0.53)),
                binned.k2.unwrap_or(R::constant(-0.0186)),
            )
        } else {
            let mut vbx = binned
                .vbx
                .unwrap_or(phi - ndep * binned.xt * binned.xt * 7.7348e-4);
            if vbx.value() > 0.0 {
                vbx = -vbx;
            }
            if binned.vbm.value() > 0.0 {
                binned.vbm = -binned.vbm;
            }
            let gamma1 = binned.gamma1.unwrap_or(ndep.sqrt() * 5.753e-12 / coxe);
            let gamma2 = binned.gamma2.unwrap_or(nsub.sqrt() * 5.753e-12 / coxe);
            let t0 = gamma1 - gamma2;
            let t1 = (phi - vbx).sqrt() - sqrt_phi;
            let t2 = (phi * (phi - binned.vbm)).sqrt() - phi;
            let k2 = t0 * t1 / (t2 * 2.0 + binned.vbm);
            (gamma2 - k2 * (phi - binned.vbm).sqrt() * 2.0, k2)
        };
        let vfb = binned.vfb.unwrap_or_else(|| match binned.vth0 {
            Some(vth0) => vth0 * sign - phi - k1 * sqrt_phi,
            _ => R::constant(-1.0),
        });
        let vth0 = binned
            .vth0
            .unwrap_or_else(|| (vfb + phi + k1 * sqrt_phi) * sign);
        let k1ox = k1 * toxe / toxm;
        let tmp = (toxe * xdep0 * EPSSI / (self.epsrox * EPS0)).sqrt();
        let theta0vb0 = theta(D::constant(binned.dsub * leff / tmp)).value;
        let theta_rout =
            binned.pdiblc1 * theta(D::constant(binned.drout * leff / tmp)).value + binned.pdiblc2;
        let abulk_cv_factor = (binned.clc / leff_cv).pow(binned.cle) + 1.0;
        let ldeb = (vtm0 * EPSSI / (ndep * (Q * 1.0e6))).sqrt() / 3.0;
        let acde = binned.acde * (ndep / 2.0e16).powf(-0.25);
        let tmp2 = factor1 * xdep0.sqrt();
        let v0 = vbi - phi;
        let t8 = binned.dvt0w * theta(D::constant(binned.dvt1w * weff * leff / tmp2)).value * v0;
        let t9 = binned.dvt0 * theta(D::constant(binned.dvt1 * leff / tmp2)).value * v0;
        let t4 = toxe * phi / (weff + binned.w0);
        let t0 = (binned.lpe0 / leff + 1.0).sqrt();
        let t3 = (binned.kt1 + binned.kt1l / leff) * rise;
        let t5 = k1ox * (t0 - 1.0) * sqrt_phi + t3;
        let vfbzb_factor = -t8 - t9 + binned.k3 * t4 + t5 - phi - k1 * sqrt_phi;

        // The stress of the length of diffusion.
        let wlod = if self.wlod.value() < 0.0 {
            R::constant(0.0)
        } else {
            self.wlod
        };
        let w_lod = w_new + wlod;
        let ku0 = lost(
            R::constant(1.0),
            [self.lku0, self.wku0, self.pku0],
            [l_new.pow(self.llodku0), w_lod.pow(self.wlodku0)],
        );
        let kvth0 = lost(
            R::constant(1.0),
            [self.lkvth0, self.wkvth0, self.pkvth0],
            [l_new.pow(self.llodvth), w_lod.pow(self.wlodvth)],
        );
        let kvth0 = (kvth0 * kvth0 + STRESS_DELTA).sqrt();
        let ku0temp = ku0 * (self.tku0 * rise + 1.0) + STRESS_DELTA;
        let half = length * 0.5;
        let inv_od_ref = (self.saref + half).recip() + (self.sbref + half).recip();
        let rho_ref = self.ku0 / ku0temp * inv_od_ref;
        let (sa, sb) = (self.diffusion_edge_a, self.diffusion_edge_b);
        let sd = self.finger_spacing.unwrap_or(self.dmcg * 2.0);
        #[expect(
            clippy::float_cmp,
            reason = "b4v5temp.c stresses a single finger, `nf` exactly 1, as it is"
        )]
        let single = fingers.value() == 1.0;
        let stressed = sa.value() > 0.0
            && sb.value() > 0.0
            && (single || (fingers.value() > 1.0 && sd.value() > 0.0));
        let (mut vth0, mut eta0, mut vsattemp) = (vth0, binned.eta0, vsattemp);
        if stressed {
            for (name, value) in [("saref", self.saref), ("sbref", self.sbref)] {
                check(value.value() <= 0.0, || {
                    format!("{name} must be positive, not {:?}", value.value())
                })?;
            }
            let kvsat = R::constant(self.kvsat.value().clamp(-1.0, 1.0));
            let mut inv_sa = R::constant(0.0);
            let mut inv_sb = R::constant(0.0);
            let mut finger = 0.0;
            while finger < fingers.value() {
                let step = (sd + length) * finger;
                inv_sa += (fingers * (sa + half + step)).recip();
                inv_sb += (fingers * (sb + half + step)).recip();
                finger += 1.0;
            }
            let inv_od_eff = inv_sa + inv_sb;
            let rho = self.ku0 / ku0temp * inv_od_eff;
            // `b4v5temp.c` scales the mobility here, then takes the unstressed one with
            // `mulu0` below: the stress leaves the mobility as it was.
            vsattemp = vsattemp * (kvsat * rho + 1.0) / (kvsat * rho_ref + 1.0);
            let offset = inv_od_eff - inv_od_ref;
            vth0 += self.kvth0 / kvth0 * offset;
            k2 += self.stk2 / kvth0.pow(self.lodk2) * offset;
            eta0 += self.steta0 / kvth0.pow(self.lodeta0) * offset;
        }
        vth0 += self.threshold_adjustment;
        let vfb = vfb + self.threshold_adjustment * sign;
        let u0temp = u0temp * self.mobility_multiplier;
        let t3 = vth0 * sign - vfb - phi;
        let vtfbphi2 = if t3.value() < 0.0 {
            R::constant(0.0)
        } else {
            t3 * 4.0
        };
        let vbsc = if k2.value() < 0.0 {
            let t0 = k1 * 0.5 / k2;
            let vbsc = (phi - t0 * t0) * 0.9;
            if vbsc.value() > -3.0 {
                R::constant(-3.0)
            } else if vbsc.value() < -30.0 {
                R::constant(-30.0)
            } else {
                vbsc
            }
        } else {
            R::constant(-30.0)
        };
        let vbsc = if vbsc.value() > binned.vbm.value() {
            binned.vbm
        } else {
            vbsc
        };
        let k2ox = k2 * toxe / toxm;
        let vfbzb = vfbzb_factor + vth0 * sign;

        // The body's resistances.
        let body_network = match self.body_resistance_model {
            BodyResistance::Zero => None,
            BodyResistance::One => {
                let conductance = |resistance: R| {
                    if resistance.value() < 1.0e-3 {
                        R::constant(1.0e3)
                    } else {
                        resistance.recip() + self.gbmin
                    }
                };
                Some(
                    [self.rbpd, self.rbps, self.rbpb, self.rbdb, self.rbsb]
                        .map(|resistance| conductance(resistance) * multiplier),
                )
            }
        };

        // The diffusions, from their areas and perimeters where given.
        let dmcg = self.dmcg - self.dmcgt;
        let dmci = self.dmci.unwrap_or(self.dmcg);
        let [source_perimeter, drain_perimeter, source_area, drain_area] =
            diffusions(fingers.value(), weff_cj, [dmcg, dmci]);
        let perimeter = |given: Option<R>, derived: R| match (given, self.perimeter_model) {
            (Some(given), Perimeter::Zero) => given,
            (Some(given), Perimeter::One) => given - weff_cj * fingers,
            (None, _) => derived,
        };
        let pseff = perimeter(self.source_perimeter, source_perimeter);
        let pdeff = perimeter(self.drain_perimeter, drain_perimeter);
        let aseff = self.source_area.unwrap_or(source_area);
        let adeff = self.drain_area.unwrap_or(drain_area);
        let resistances = self.resistances()?;
        let series = resistances.map(series_value);

        // The junctions, `diomod` 1, and their trap-assisted tunneling.
        let gate_edge = weff_cj * fingers;
        let saturation = |area: R, perimeter: R, densities: [R; 3]| {
            if area.value() <= 0.0 && perimeter.value() <= 0.0 {
                R::constant(1.0e-14)
            } else {
                area * densities[0] + perimeter * densities[1] + gate_edge * densities[2]
            }
        };
        let source_saturation = saturation(aseff, pseff, source_densities);
        let drain_saturation = saturation(adeff, pdeff, drain_densities);
        let njtstemp = self.njts * (self.tnjts * rise + 1.0);
        let njtsswtemp = self.njtssw * (self.tnjtssw * rise + 1.0);
        let njtsswgtemp = self.njtsswg * (self.tnjtsswg * rise + 1.0);
        let t7 = eg0 / vtm * rise;
        let tunneling = |[bottom, sidewall, gate]: [R; 3],
                         [xts_bottom, xts_sidewall, xts_gate]: [R; 3],
                         [vts_bottom, vts_sidewall, vts_gate]: [R; 3],
                         [area, perimeter]: [R; 2]| {
            [
                (
                    dexp(D::constant(xts_bottom * t7)).value * area * bottom,
                    vts_bottom,
                    vtm0 * njtstemp,
                ),
                (
                    dexp(D::constant(xts_sidewall * t7)).value * perimeter * sidewall,
                    vts_sidewall,
                    vtm0 * njtsswtemp,
                ),
                (
                    dexp(D::constant(xts_gate * t7)).value * gate_edge * gate,
                    vts_gate,
                    vtm0 * njtsswgtemp,
                ),
            ]
        };
        let junction = |saturation_current: R, emission: R, ijth: R, tunneling| {
            let slope = vtm * emission;
            let vjm = slope * (ijth / saturation_current + 1.0).ln();
            Junction {
                saturation_current,
                slope,
                vjm,
                current_at_vjm: saturation_current * (vjm / slope).exp(),
                tunneling,
            }
        };
        let vts_source = [self.vtss, self.vtssws, self.vtsswgs];
        let vts_drain = [
            self.vtsd.unwrap_or(self.vtss),
            self.vtsswd.unwrap_or(self.vtssws),
            self.vtsswgd.unwrap_or(self.vtsswgs),
        ];
        for (name, value) in ["vtss", "vtssws", "vtsswgs"]
            .into_iter()
            .zip(vts_source)
            .chain(["vtsd", "vtsswd", "vtsswgd"].into_iter().zip(vts_drain))
        {
            check(value.value() < 0.0, || {
                format!("{name} must be at least 0, not {:?}", value.value())
            })?;
        }
        let source_junction = junction(
            source_saturation,
            source_emission,
            ijthsfwd,
            tunneling(
                [self.jtss, self.jtssws, self.jtsswgs],
                [self.xtss, self.xtssws, self.xtsswgs],
                vts_source,
                [aseff, pseff],
            ),
        );
        let drain_junction = junction(
            drain_saturation,
            drain_emission,
            ijthdfwd,
            tunneling(
                [
                    self.jtsd.unwrap_or(self.jtss),
                    self.jtsswd.unwrap_or(self.jtssws),
                    self.jtsswgd.unwrap_or(self.jtsswgs),
                ],
                [
                    self.xtsd.unwrap_or(self.xtss),
                    self.xtsswd.unwrap_or(self.xtssws),
                    self.xtsswgd.unwrap_or(self.xtsswgs),
                ],
                vts_drain,
                [adeff, pdeff],
            ),
        );

        self.checked(
            &binned,
            [leff, weff],
            [ndep, nsub, phi, ngate],
            [u0temp, vsattemp],
        )?;
        if self.check_parameters {
            // ngspice's `paramchk = 1` clamps what it warns of.
            if binned.a2.value() < 0.01 {
                binned.a2 = R::constant(0.01);
            } else if binned.a2.value() > 1.0 {
                binned.a2 = R::constant(1.0);
                binned.a1 = R::constant(0.0);
            }
            if binned.prwg.value() < 0.0 {
                binned.prwg = R::constant(0.0);
            }
            if binned.rdsw.value() < 0.0 {
                binned.rdsw = R::constant(0.0);
                rds0 = R::constant(0.0);
            }
            if rds0.value() < 0.0 {
                rds0 = R::constant(0.0);
            }
            if rdswmin.value() < 0.0 {
                rdswmin = R::constant(0.0);
            }
        }
        let external =
            (self.drain_source_resistance_model == DrainSourceResistance::One).then(|| {
                let t10 = binned.prt * rise;
                let at_least_zero = |resistance: R| {
                    if resistance.value() < 0.0 {
                        R::constant(0.0)
                    } else {
                        resistance
                    }
                };
                let sheet = [self.drain_squares, self.source_squares].map(|squares| match squares
                    .map(|squares| self.rsh * squares)
                {
                    Some(resistance) if resistance.value() > 0.0 => resistance.recip(),
                    _ => R::constant(1.0e3),
                });
                External {
                    minimums: [self.rdwmin, self.rswmin]
                        .map(|minimum| at_least_zero(minimum + t10) / pow_weff_wr),
                    resistances: [binned.rdw, binned.rsw]
                        .map(|resistance| at_least_zero(resistance + t10) / pow_weff_wr),
                    vfbsd: if ngate.value() > 0.0 {
                        vtm0 * (ngate / nsd).ln()
                    } else {
                        R::constant(0.0)
                    },
                    prwg: binned.prwg,
                    prwb: binned.prwb,
                    sheet,
                }
            });
        if external.is_some() {
            // `b4v5ld.c` takes no resistance within the drain current at `rdsmod` 1.
            rds0 = R::constant(0.0);
            rdswmin = R::constant(0.0);
        }

        // The overlaps, their defaults from the oxide.
        let dlc_given = self.dlc.filter(|dlc| dlc.value() > 0.0);
        let overlap = |given: Option<R>, lightly: R| {
            given.unwrap_or_else(|| {
                dlc_given.map_or(self.xj * coxe * 0.6, |dlc| dlc * coxe - lightly)
            })
        };
        let cgso_model = overlap(self.cgso, self.cgsl);
        let cgdo_model = overlap(self.cgdo, self.cgdl);
        let cgbo_model = self.cgbo.unwrap_or(dwc_model * coxe * 2.0);
        let at_least_kappa = |kappa: R| {
            if kappa.value() < 0.02 {
                R::constant(0.02)
            } else {
                kappa
            }
        };
        let overlaps = Overlaps {
            cgso: (cgso_model + binned.cf) * weff_cv,
            cgdo: (cgdo_model + binned.cf) * weff_cv,
            cgsl: binned.cgsl * weff_cv,
            cgdl: binned.cgdl * weff_cv,
            ckappas: at_least_kappa(binned.ckappas),
            ckappad: at_least_kappa(binned.ckappad),
            cgbo: cgbo_model * leff_cv * fingers,
            fingers,
        };

        // The junctions' depletion, their capacitances and potentials at the
        // temperature, the drain's defaulting to the source's.
        let heated_capacitance = |capacitance: R, coefficient: R| {
            let t0 = coefficient * delta_temperature;
            if t0.value() >= -1.0 {
                capacitance * (t0 + 1.0)
            } else {
                R::constant(0.0)
            }
        };
        let heated_potential = |potential: R, coefficient: R| {
            let potential = if potential.value() < 0.1 {
                R::constant(0.1)
            } else {
                potential
            };
            let heated = potential - coefficient * delta_temperature;
            if heated.value() <= 0.01 {
                R::constant(0.01)
            } else {
                heated
            }
        };
        let cjswgs = self.cjswgs.unwrap_or(self.cjsws);
        let pbswgs = self.pbswgs.unwrap_or(self.pbsws);
        let mjswgs = self.mjswgs.unwrap_or(self.mjsws);
        let depletion = |[bottom, sidewall, gate]: [R; 3],
                         [pb, pbsw, pbswg]: [R; 3],
                         gradings: [R; 3],
                         [area, perimeter]: [R; 2]| Depletion {
            capacitances: [
                heated_capacitance(bottom, self.tcj) * area,
                heated_capacitance(sidewall, self.tcjsw) * perimeter,
                heated_capacitance(gate, self.tcjswg) * gate_edge,
            ],
            potentials: [
                heated_potential(pb, self.tpb),
                heated_potential(pbsw, self.tpbsw),
                heated_potential(pbswg, self.tpbswg),
            ],
            gradings,
        };
        let source_depletion = depletion(
            [self.cjs, self.cjsws, cjswgs],
            [self.pbs, self.pbsws, pbswgs],
            [self.mjs, self.mjsws, mjswgs],
            [aseff, pseff],
        );
        let drain_depletion = depletion(
            [
                self.cjd.unwrap_or(self.cjs),
                self.cjswd.unwrap_or(self.cjsws),
                self.cjswgd.unwrap_or(cjswgs),
            ],
            [
                self.pbd.unwrap_or(self.pbs),
                self.pbswd.unwrap_or(self.pbsws),
                self.pbswgd.unwrap_or(pbswgs),
            ],
            [
                self.mjd.unwrap_or(self.mjs),
                self.mjswd.unwrap_or(self.mjsws),
                self.mjswgd.unwrap_or(mjswgs),
            ],
            [adeff, pdeff],
        );

        let noise = Noise {
            flicker_model: self.flicker_noise_model,
            thermal_model: self.thermal_noise_model,
            noia: self
                .noia
                .map_or(by_channel(self.channel, 6.25e41, 6.188e40), Real::value),
            noib: self
                .noib
                .map_or(by_channel(self.channel, 3.125e26, 1.5e25), Real::value),
            noic: self.noic.value(),
            em: self.em.value(),
            ef: self.ef.value(),
            af: self.af.value(),
            kf: self.kf.value(),
            ntnoi: self.ntnoi.value(),
            tnoia: self.tnoia.value(),
            tnoib: self.tnoib.value(),
            rnoia: self.rnoia.value(),
            rnoib: self.rnoib.value(),
            length: (leff - self.lintnoi * 2.0).value(),
        };
        Ok(PreparedBsim4 {
            external,
            overlaps,
            resistances,
            noise,
            temperature: temperature.value(),
            drain_depletion,
            source_depletion,
            core: Core {
                sign,
                toxe,
                toxp,
                coxe,
                coxp,
                factor1,
                vtm,
                temperature_rise: rise,
                leff,
                weff,
                weff_cj,
                leff_cv,
                weff_cv,
                fingers,
                xpart: self.charge_partition,
                abulk_cv_factor,
                vfbzb,
                ldeb,
                acde,
                phi,
                sqrt_phi,
                xdep0,
                vbi,
                cdep0,
                litl,
                k1,
                k1ox,
                k2ox,
                mstar,
                voffcbn,
                theta0vb0,
                theta_rout,
                rds0,
                rdswmin,
                ua,
                ub,
                uc,
                ud,
                u0temp,
                vsattemp,
                tfactor,
                vth0,
                vfb,
                eta0,
                vbsc,
                vtfbphi2,
                ngate,
                pditsl: self.pditsl,
                lambda: self
                    .lambda
                    .is_some_and(|lambda| lambda.value() > 0.0)
                    .then_some(binned.lambda),
                binned,
            },
            drain_junction,
            source_junction,
            series,
            body_network,
            multiplier,
        })
    }

    /// Fails on what `b4v5check.c` stops on in the binned parameters at the effective
    /// length and width, the depletion's and the gate's dopings, the surface potential
    /// and the mobility and saturation velocity at the temperature.
    fn checked(
        &self,
        binned: &Binned<R>,
        [leff, weff]: [R; 2],
        [ndep, nsub, phi, ngate]: [R; 4],
        [u0temp, vsattemp]: [R; 2],
    ) -> Result<(), ParameterError> {
        let leff = leff.value();
        let weff = weff.value();
        for (name, value) in [("lpe0", binned.lpe0), ("lpeb", binned.lpeb)] {
            check(value.value() < -leff, || {
                format!(
                    "{name}, {:?}, must be at least the negative of the effective length, {leff:?}",
                    value.value()
                )
            })?;
        }
        check(self.lintnoi.value() > leff / 2.0, || {
            format!(
                "lintnoi, {:?}, must be at most half the effective length, {leff:?}",
                self.lintnoi.value()
            )
        })?;
        for (name, value) in [
            ("ndep", ndep),
            ("phi", phi),
            ("nsub", nsub),
            ("xj", binned.xj),
            ("u0 at the temperature", u0temp),
            ("vsat at the temperature", vsattemp),
            ("pclm", binned.pclm),
        ] {
            check(value.value() <= 0.0, || {
                format!("{name} must be positive, not {:?}", value.value())
            })?;
        }
        check(ngate.value() < 0.0 || ngate.value() > 1.0e25, || {
            format!(
                "ngate must lie between 0 and 1e31 per cubic meter, not {:?}",
                binned.ngate.value()
            )
        })?;
        for (name, value) in [
            ("dvt1", binned.dvt1),
            ("dvt1w", binned.dvt1w),
            ("dsub", binned.dsub),
            ("delta", binned.delta),
            ("drout", binned.drout),
            ("fprout", binned.fprout),
            ("pdits", binned.pdits),
            ("pditsl", self.pditsl),
            ("clc", binned.clc),
        ] {
            check(value.value() < 0.0, || {
                format!("{name} must be at least 0, not {:?}", value.value())
            })?;
        }
        for (name, value) in [("w0", binned.w0), ("b1", binned.b1)] {
            #[expect(
                clippy::float_cmp,
                reason = "b4v5check.c stops on this exact equality, a division by zero"
            )]
            let vanishing = value.value() == -weff;
            check(vanishing, || {
                format!("{name} must not be the negative of the effective width, {weff:?}")
            })?;
        }
        Ok(())
    }
}
