//! A BSIM4 transistor's channel at one bias: its threshold, its drain current, the
//! impact ionization's substrate current and the gate-induced leakages, as ngspice 47's
//! `b4v5ld.c` computes them.
//!
//! Part of jolty's BSIM4, derived from BSIM4.5.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2005 Regents of the University of California. All rights reserved. Used
//! under the BSIM4 terms of use, which the repository's `NOTICE` reproduces.

use super::D;
use super::prepare::Core;
use crate::Real;
use crate::kernel::bsim3::{
    EPSSI, EXP_THRESHOLD, MAX_EXP, MIN_EXP, NOISE_CHARGE, Q, above_half, above_tenth,
    factor_above_tenth, reciprocal_above_tenth,
};

/// The exponent of `b4v5ld.c`'s smoothing of the source end's velocity limit, `2 · MM`
/// with its `MM` of 3.
const SOURCE_END_EXPONENT: f64 = 6.0;
/// The least argument of `b4v5ld.c`'s gate-induced leakage's exponential, below whose
/// negative it holds the leakage at its `3.720075976e-44` times its prefactor.
const GIDL_EXPONENT: f64 = 100.0;
/// `e^−100`, as `b4v5ld.c` writes it for the gate-induced leakage.
const GIDL_MIN_EXP: f64 = 3.720_075_976e-44;

/// What a channel of all its fingers carries at one bias, in the frame whose drain lies
/// at or above its source, each with its derivatives with respect to the frame's gate-source,
/// drain-source and body-source voltages.
pub(super) struct Operating<R: Real> {
    /// The drain current.
    pub(super) ids: D<R>,
    /// The impact ionization's substrate current, from the drain into the body.
    pub(super) isub: D<R>,
    pub(super) vth: D<R>,
    pub(super) vgs_eff: D<R>,
    pub(super) vdsat: D<R>,
    /// The body's effective voltage, the subthreshold swing and `Abulk0`, which the
    /// charges read.
    pub(super) vbseff: D<R>,
    pub(super) n: D<R>,
    pub(super) abulk0: D<R>,
    /// What the noise reads of one finger: the mobility, the effective gate-source and
    /// drain-source voltages, `EsatL`, `Abulk` after and before `keta`, the oxide's
    /// capacitance through the inversion layer, the source and drain resistance, the
    /// current over `Vdseff` and `nstar`.
    pub(super) ueff: D<R>,
    pub(super) vgsteff: D<R>,
    pub(super) vdseff: D<R>,
    pub(super) esat_l: D<R>,
    pub(super) abulk: D<R>,
    pub(super) abulk_unscaled: D<R>,
    pub(super) coxeff: D<R>,
    pub(super) rds: D<R>,
    pub(super) ids_over_vds: D<R>,
    pub(super) nstar: D<R>,
}

/// `b4v5temp.c`'s and `b4v5ld.c`'s `T1 / ((T1 − 1)² + 2·T1·MIN_EXP)` of `T1 = e^x`,
/// held at `1 / (MAX_EXP − 2)` from [`EXP_THRESHOLD`] on.
pub(super) fn theta<R: Real>(x: D<R>) -> D<R> {
    if x.value.value() < EXP_THRESHOLD {
        let t1 = x.exp();
        let t2 = t1 - 1.0;
        t1 / (t2 * t2 + t1 * (2.0 * MIN_EXP))
    } else {
        D::number(1.0 / (MAX_EXP - 2.0))
    }
}

impl<R: Real> Core<R> {
    /// The gate's voltage less the poly gate's depletion, `BSIM4v5polyDepletion` at the
    /// gate-source voltage `vgs`.
    fn poly_depletion(&self, vgs: D<R>) -> D<R> {
        let phi = self.vfb + self.phi;
        let ngate = self.ngate.value();
        if ngate > 1.0e18 && ngate < 1.0e25 && vgs.value.value() > phi.value() {
            let t1 = self.ngate * (1.0e6 * NOISE_CHARGE * EPSSI) / (self.coxe * self.coxe);
            let t8 = vgs.plus(-phi);
            let t4 = (t8.divide(t1) * 2.0 + 1.0).sqrt();
            let t2 = t8 * 2.0 / (t4 + 1.0);
            let t3 = (t2 * t2 * 0.5).divide(t1);
            let t7 = 1.12 - t3 - 0.05;
            let t6 = (t7 * t7 + 0.224).sqrt();
            let t5 = 1.12 - (t7 + t6) * 0.5;
            vgs - t5
        } else {
            vgs
        }
    }

    /// The channel at the gate-source, drain-source and body-source voltages of the frame
    /// whose drain lies at or above its source.
    #[expect(
        clippy::too_many_lines,
        clippy::similar_names,
        reason = "one pass through b4v5ld.c's drain current, in its order, by its names"
    )]
    pub(super) fn evaluate(&self, [vgs, vds, vbs]: [D<R>; 3]) -> Operating<R> {
        let p = &self.binned;
        let c = D::constant;
        let leff = self.leff;
        let vtm = self.vtm;

        // The body's effective voltage, kept below 0.95 φ.
        let t0 = (vbs - c(self.vbsc)) - 0.001;
        let t1 = (t0 * t0 - c(self.vbsc) * 0.004).sqrt();
        let vbseff = if t0.value.value() >= 0.0 {
            (t0 + t1) * 0.5 + c(self.vbsc)
        } else {
            let t2 = (t1 - t0).recip() * -0.002;
            (t2 + 1.0).scale(self.vbsc)
        };
        let t9 = self.phi * 0.95;
        let t0 = c(t9) - vbseff - 0.001;
        let t1 = (t0 * t0 + c(t9 * 0.004)).sqrt();
        let vbseff = c(t9) - (t0 + t1) * 0.5;
        let phis = c(self.phi) - vbseff;
        let sqrt_phis = phis.sqrt();
        let xdep = sqrt_phis.scale(self.xdep0 / self.sqrt_phi);

        // The threshold.
        let t3 = xdep.sqrt();
        let v0 = self.vbi - self.phi;
        let lt1 = (t3 * above_half(vbseff.scale(p.dvt2))).scale(self.factor1);
        let ltw = (t3 * above_half(vbseff.scale(p.dvt2w))).scale(self.factor1);
        let theta0 = theta(c(p.dvt1 * leff) / lt1);
        let delt_vth = theta0.scale(p.dvt0 * v0);
        let t2 = theta(c(p.dvt1w * self.weff * leff) / ltw).scale(p.dvt0w * v0);
        let t0 = (p.lpe0 / leff + 1.0).sqrt();
        let t1 = (vbseff.scale(p.kt2) + c(p.kt1 + p.kt1l / leff)).scale(self.temperature_rise)
            + c(self.k1ox * (t0 - 1.0) * self.sqrt_phi);
        let vth_narrow_w = self.toxe * self.phi / (self.weff + p.w0);
        let t3 = vbseff.scale(p.etab) + c(self.eta0);
        let t3 = if t3.value.value() < 1.0e-4 {
            (2.0e-4 - t3) / (3.0 - t3 * 2.0e4)
        } else {
            t3
        };
        let dibl_sft = (t3 * vds).scale(self.theta0vb0);
        let lpe_vb = (p.lpeb / leff + 1.0).sqrt();
        let mut vth = (sqrt_phis.scale(self.k1ox) - c(self.k1 * self.sqrt_phi)).scale(lpe_vb)
            + c(self.vth0 * self.sign)
            - vbseff.scale(self.k2ox)
            - delt_vth
            - t2
            + (vbseff.scale(p.k3b) + c(p.k3)).scale(vth_narrow_w)
            + t1
            - dibl_sft;

        // The subthreshold swing.
        let tmp1 = xdep.recip() * EPSSI;
        let tmp2 = tmp1.scale(p.nfactor);
        let tmp3 = vbseff.scale(p.cdscb) + vds.scale(p.cdscd) + c(p.cdsc);
        let tmp4 = (tmp2 + tmp3 * theta0).plus(p.cit).divide(self.coxe);
        let n = above_half(tmp4);

        // The pocket implant's shift of the threshold.
        if p.dvtp0.value() > 0.0 {
            let t0 = -vds.scale(p.dvtp1);
            let t2 = if t0.value.value() < -EXP_THRESHOLD {
                D::number(MIN_EXP)
            } else {
                t0.exp()
            };
            let t3 = (t2 + 1.0).scale(p.dvtp0).plus(leff);
            let t4 = (c(leff) / t3).ln().scale(vtm);
            vth = vth - n * t4;
        }

        // The effective gate-source voltage, from weak to strong inversion.
        let vgs_eff = self.poly_depletion(vgs);
        let vgst = vgs_eff - vth;
        let t0 = n.scale(vtm);
        let t1 = vgst.scale(self.mstar);
        let t2 = t1 / t0;
        let t10 = if t2.value.value() > EXP_THRESHOLD {
            t1
        } else if t2.value.value() < -EXP_THRESHOLD {
            n.scale(vtm * MIN_EXP.ln_1p())
        } else {
            n * t2.exp().ln_1p().scale(vtm)
        };
        let t1 = c(self.voffcbn) - vgst.scale(-self.mstar + 1.0);
        let t2 = t1 / t0;
        let t9 = if t2.value.value() < -EXP_THRESHOLD {
            n.scale(self.coxe * MIN_EXP / self.cdep0).plus(self.mstar)
        } else if t2.value.value() > EXP_THRESHOLD {
            n.scale(self.coxe * MAX_EXP / self.cdep0).plus(self.mstar)
        } else {
            (n * t2.exp())
                .scale(self.coxe / self.cdep0)
                .plus(self.mstar)
        };
        let vgsteff = t10 / t9;

        // The effective width and the source and drain resistance.
        let t9 = sqrt_phis - c(self.sqrt_phi);
        let weff = c(self.weff) - (vgsteff.scale(p.dwg) + t9.scale(p.dwb)) * 2.0;
        let weff = if weff.value.value() < 2.0e-8 {
            (4.0e-8 - weff) * 2.0e-8 / (6.0e-8 - weff * 2.0)
        } else {
            weff
        };
        let t0 = vgsteff.scale(p.prwg) + 1.0;
        let t2 = t0.recip() + t9.scale(p.prwb);
        let t3 = t2 + (t2 * t2 + 0.01).sqrt();
        let rds = t3.scale(self.rds0 * 0.5).plus(self.rdswmin);

        // The bulk charge effect.
        let t9 = (sqrt_phis.recip() * 0.5).scale(self.k1ox * lpe_vb);
        let t1 = t9.plus(self.k2ox - p.k3b * vth_narrow_w);
        let t9 = xdep.scale(p.xj).sqrt();
        let t5 = c(leff) / (t9 * 2.0).plus(leff);
        let t2 = t5.scale(p.a0).plus(p.b0 / (self.weff + p.b1));
        let t7 = t5 * t5 * t5;
        let abulk0 = t1 * t2 + 1.0;
        let t8 = t7.scale(p.ags * p.a0);
        let abulk = abulk0 - t1 * t8 * vgsteff;
        let abulk0 = above_tenth(abulk0);
        let abulk_unscaled = above_tenth(abulk);
        let t0_keta = reciprocal_above_tenth(vbseff.scale(p.keta));
        let abulk = abulk_unscaled * t0_keta;

        // The mobility, `mobmod` 0.
        let t0 = vgsteff + vth + vth;
        let t2 = vbseff.scale(self.uc).plus(self.ua);
        let t3 = t0.divide(self.toxe);
        let t6 = (vth * vth).scale(self.ud) / (t3 * t3);
        let t5 = t3 * (t2 + t3.scale(self.ub)) + t6;
        let denominator = if t5.value.value() >= -0.8 {
            t5 + 1.0
        } else {
            (t5 + 0.6) / (t5 * 10.0 + 7.0)
        };
        let ueff = denominator.recip().scale(self.u0temp);

        // The saturation voltage.
        let wv_cox = weff.scale(self.vsattemp * self.coxe);
        let wv_cox_rds = wv_cox * rds;
        let esat = ueff.recip().scale(self.vsattemp * 2.0);
        let mut esat_l = esat.scale(leff);
        let a1 = p.a1.value();
        let lambda = if a1 == 0.0 {
            c(p.a2)
        } else if a1 > 0.0 {
            let t0 = -p.a2 + 1.0;
            let t1 = c(t0) - vgsteff.scale(p.a1) - 0.0001;
            let t2 = (t1 * t1 + c(t0 * 0.0004)).sqrt();
            c(p.a2 + t0) - (t1 + t2) * 0.5
        } else {
            let t1 = vgsteff.scale(p.a1).plus(p.a2) - 0.0001;
            let t2 = (t1 * t1 + c(p.a2 * 0.0004)).sqrt();
            (t1 + t2) * 0.5
        };
        let vgst2vtm = vgsteff.plus(vtm * 2.0);
        #[expect(
            clippy::float_cmp,
            reason = "b4v5ld.c takes the simpler root at exactly no resistance and λ = 1"
        )]
        let simple = rds.value.value() == 0.0 && lambda.value.value() == 1.0;
        let vdsat = if simple {
            esat_l * vgst2vtm / (abulk * esat_l + vgst2vtm)
        } else {
            let t9 = abulk * wv_cox_rds;
            let t7 = vgst2vtm * t9;
            let t6 = vgst2vtm * wv_cox_rds;
            let t0 = abulk * (t9 - 1.0 + lambda.recip()) * 2.0;
            let t1 = vgst2vtm * (lambda.recip() * 2.0 - 1.0) + abulk * esat_l + t7 * 3.0;
            let t2 = vgst2vtm * (esat_l + t6 * 2.0);
            let t3 = (t1 * t1 - t0 * t2 * 2.0).sqrt();
            (t1 - t3) / t0
        };

        // The effective drain-source voltage.
        let delta = p.delta;
        let t1 = vdsat - vds - c(delta);
        let t2 = (t1 * t1 + vdsat.scale(delta * 4.0)).sqrt();
        let mut vdseff = if t1.value.value() >= 0.0 {
            vdsat - (t1 + t2) * 0.5
        } else {
            let t4 = c(delta * 2.0) / (t2 - t1);
            vdsat * (1.0 - t4)
        };
        if vds.value.value() == 0.0 {
            let [_, drain, _] = vdseff.gradient;
            vdseff = D {
                value: R::constant(0.0),
                gradient: [R::constant(0.0), drain, R::constant(0.0)],
            };
        }
        if vdseff.value.value() > vds.value.value() {
            vdseff = vds;
        }
        let diff_vds = vds - vdseff;

        // The velocity overshoot.
        if let Some(overshoot) = self.lambda {
            let t1 = ueff.scale(leff);
            let t2 = t1.recip().scale(overshoot);
            let t5 = esat.scale(self.litl).recip();
            let t6 = diff_vds * t5 + 1.0;
            let t7 = (t6 * t6 + 1.0).recip() * 2.0;
            let t8 = 1.0 - t7;
            esat_l = esat_l * (t2 * t8 + 1.0);
        }

        // The drain current's saturation voltage of the output resistance.
        let tmp4 = 1.0 - (abulk * vdsat / vgst2vtm) * 0.5;
        let t9 = wv_cox_rds * vgsteff;
        let t0 = esat_l + vdsat + t9 * tmp4 * 2.0;
        let t1 = lambda.recip() * 2.0 - 1.0 + wv_cox_rds * abulk;
        let vasat = t0 / t1;

        // The linear current, with the inversion layer's thickness.
        let t0 = (vgsteff.plus(self.vtfbphi2)).divide(self.toxp * 2.0e8);
        let tmp3 = (t0.ln() * 0.7).exp();
        let tcen = (tmp3 + 1.0).recip() * 1.9e-9;
        let coxeff = (tcen.scale(self.coxp) + EPSSI)
            .recip()
            .scale(self.coxp * EPSSI);
        let coxeff_wov_l = (coxeff * weff).divide(leff);
        let beta = ueff * coxeff_wov_l;
        let abov_vgst2vtm = abulk / vgst2vtm;
        let fgche1 = vgsteff * (1.0 - vdseff * abov_vgst2vtm * 0.5);
        let fgche2 = vdseff / esat_l + 1.0;
        let gche = beta * fgche1 / fgche2;
        let idl = gche / (gche * rds + 1.0);

        // The output resistance: channel-length modulation, barrier lowering, the
        // threshold's drain-induced shift and the substrate current.
        let fp = if p.fprout.value() <= 0.0 {
            D::number(1.0)
        } else {
            (vgst2vtm.recip().scale(p.fprout * leff.sqrt()) + 1.0).recip()
        };
        let t9 = (vgsteff / esat_l).scale(p.pvag);
        let pvag_term = factor_above_tenth(t9, false);
        let (vaclm, cclm) = if p.pclm.value() > MIN_EXP && diff_vds.value.value() > 1.0e-10 {
            let t0 = rds * idl + 1.0;
            let t1 = vdsat / esat + c(leff);
            let cclm = (fp * pvag_term * t0 * t1).divide(p.pclm * self.litl);
            (cclm * diff_vds, cclm)
        } else {
            (D::number(MAX_EXP), D::number(MAX_EXP))
        };
        let vadibl = if self.theta_rout.value() > MIN_EXP {
            let t8 = abulk * vdsat;
            let t0 = vgst2vtm * t8;
            let t1 = vgst2vtm + t8;
            let vadibl = (vgst2vtm - t0 / t1).divide(self.theta_rout);
            vadibl * reciprocal_above_tenth(vbseff.scale(p.pdiblcb)) * pvag_term
        } else {
            D::number(MAX_EXP)
        };
        let va = vasat + vaclm;
        let t0 = vds.scale(p.pditsd);
        let t1 = if t0.value.value() > EXP_THRESHOLD {
            D::number(MAX_EXP)
        } else {
            t0.exp()
        };
        let vadits = if p.pdits.value() > MIN_EXP {
            let t2 = self.pditsl * leff + 1.0;
            (t1.scale(t2) + 1.0).divide(p.pdits) * fp
        } else {
            D::number(MAX_EXP)
        };
        let vascbe = if p.pscbe2.value() > 0.0 {
            let threshold = p.pscbe1 * self.litl / EXP_THRESHOLD;
            if diff_vds.value.value() > threshold.value() {
                let t0 = diff_vds.recip().scale(p.pscbe1 * self.litl);
                t0.exp().scale(leff / p.pscbe2)
            } else {
                c(leff * MAX_EXP / p.pscbe2)
            }
        } else {
            D::number(MAX_EXP)
        };
        let idsa = idl * (diff_vds / vadibl + 1.0) * (diff_vds / vadits + 1.0);
        let idsa = idsa * ((va / vasat).ln() / cclm + 1.0);

        // The impact ionization's substrate current.
        let tmp = p.alpha0 + p.alpha1 * leff;
        let isub = if tmp.value() <= 0.0 || p.beta0.value() <= 0.0 {
            D::number(0.0)
        } else {
            let t2 = tmp / leff;
            let t1 = if diff_vds.value.value() > (p.beta0 / EXP_THRESHOLD).value() {
                let t0 = -diff_vds.recip().scale(p.beta0);
                (diff_vds * t0.exp()).scale(t2)
            } else {
                diff_vds.scale(t2 * MIN_EXP)
            };
            t1 * idsa * vdseff
        };

        // The drain current, and the source end's velocity limit.
        let ids = idsa * (diff_vds / vascbe + 1.0);
        let mut cdrain = ids * vdseff;
        if let Some(tfactor) = self.tfactor {
            let t11 = (coxeff_wov_l * vgsteff).scale(leff).recip();
            let vs = cdrain * t11;
            let t1 = vs.divide(p.vtl * tfactor);
            if t1.value.value() > 0.0 {
                let t2 = (t1.ln() * SOURCE_END_EXPONENT).exp() + 1.0;
                let fsevl = (t2.ln() / SOURCE_END_EXPONENT).exp().recip();
                cdrain = cdrain * fsevl;
            }
        }
        // `b4v5ld.c` computes one finger's currents and multiplies them by `nf`.
        Operating {
            ids: cdrain.scale(self.fingers),
            isub: isub.scale(self.fingers),
            vth,
            vgs_eff,
            vdsat,
            vbseff,
            n,
            abulk0: abulk0 * t0_keta,
            ueff,
            vgsteff,
            vdseff,
            esat_l,
            abulk,
            abulk_unscaled,
            coxeff,
            rds,
            ids_over_vds: ids,
            nstar: (xdep.recip() * EPSSI)
                .plus(self.coxe + p.cit)
                .scale(vtm / Q),
        }
    }

    /// The gate-induced drain and source leakages of every finger, from the drain and
    /// from the source into the body, at the gate-source, drain-source and body-source
    /// voltages `voltages`, whichever end acts as the source.
    pub(super) fn gate_induced_leakages(&self, [vgs, vds, vbs]: [D<R>; 3]) -> [D<R>; 2] {
        let p = &self.binned;
        let vgd = vgs - vds;
        let vbd = vbs - vds;
        let t0 = self.toxe * 3.0;
        let leakage = |t1: D<R>, vb: D<R>| {
            if p.agidl.value() <= 0.0
                || p.bgidl.value() <= 0.0
                || t1.value.value() <= 0.0
                || p.cgidl.value() <= 0.0
                || vb.value.value() > 0.0
            {
                return D::number(0.0);
            }
            let t2 = t1.recip().scale(p.bgidl);
            let prefactor = p.agidl * self.weff_cj;
            let current = if t2.value.value() < GIDL_EXPONENT {
                (t1 * (-t2).exp()).scale(prefactor)
            } else {
                t1.scale(prefactor * GIDL_MIN_EXP)
            };
            let t5 = -(vb * vb * vb);
            current * t5 / t5.plus(p.cgidl)
        };
        let drain = (vds - self.poly_depletion(vgs)).plus(-p.egidl).divide(t0);
        let source = (-vds - self.poly_depletion(vgd)).plus(-p.egidl).divide(t0);
        [leakage(drain, vbd), leakage(source, vbs)].map(|current| current.scale(self.fingers))
    }
}
