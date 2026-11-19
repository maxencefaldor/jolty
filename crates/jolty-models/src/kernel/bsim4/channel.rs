//! A BSIM4 transistor's channel at one bias: its threshold, its drain current, the
//! impact ionization's substrate current and the gate-induced leakages, as ngspice 47's
//! `b4v5ld.c` computes them.
//!
//! Part of jolty's BSIM4, derived from BSIM4.5.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2005 Regents of the University of California. All rights reserved. Used
//! under the BSIM4 terms of use, which the repository's `NOTICE` reproduces.

use super::prepare::Core;
use crate::Real;
use crate::kernel::bsim::{
    D, EPSSI, EXP_THRESHOLD, MAX_EXP, MIN_EXP, NOISE_CHARGE, Q, above_dibl_floor, above_fifth,
    above_half, above_tenth, effective_drain, effective_width, factor_above_tenth,
    impact_ionization_factor, inversion_centroid, non_saturation_excess, poly_depletion,
    reciprocal_above_tenth, saturation_voltage, smooth_floor, smooth_positive,
    substrate_early_voltage,
};

/// The exponent of `b4v5ld.c`'s smoothing of the source end's velocity limit, `2 · MM`
/// with its `MM` of 3.
const SOURCE_END_EXPONENT: f64 = 6.0;
/// The least argument of `b4v5ld.c`'s gate-induced leakage's exponential, below whose
/// negative it holds the leakage at its `3.720075976e-44` times its prefactor.
const GIDL_EXPONENT: f64 = 100.0;
/// Lower exponential floor `e^−100` used by `b4v5ld.c` for gate-induced leakage.
const GIDL_MIN_EXP: f64 = 3.720_075_976e-44;

/// Multi-finger channel evaluation state at one bias in the forward frame (`vds >= 0`),
/// carrying values and derivatives with respect to frame gate-source, drain-source, and
/// body-source voltages.
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
    /// Single-finger quantities read by noise and gate intrinsic-input resistance: mobility,
    /// effective gate-source and drain-source voltages, `EsatL`, and `Abulk` before and
    /// after `keta`. Subsequent fields store inversion-layer oxide capacitance, series
    /// resistance, `β`, pre-velocity-limit `Ids / Vdseff`, and `nstar`.
    pub(super) ueff: D<R>,
    pub(super) vgsteff: D<R>,
    pub(super) vdseff: D<R>,
    pub(super) esat_l: D<R>,
    pub(super) abulk: D<R>,
    pub(super) abulk_unscaled: D<R>,
    pub(super) coxeff: D<R>,
    pub(super) rds: D<R>,
    pub(super) beta: D<R>,
    pub(super) ids_over_vds: D<R>,
    pub(super) nstar: D<R>,
}

/// Evaluates `T1 / ((T1 − 1)² + 2·T1·MIN_EXP)` for `T1 = e^x`, held at `1 / (MAX_EXP − 2)`
/// from [`EXP_THRESHOLD`] upward, matching `b4v5temp.c` and `b4v5ld.c`.
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
    /// Computes gate-source voltage `vgs` reduced by polysilicon gate depletion, matching
    /// `BSIM4v5polyDepletion`.
    fn poly_depletion(&self, vgs: D<R>) -> D<R> {
        poly_depletion(
            vgs,
            self.ngate,
            self.vfb + self.phi,
            self.coxe,
            NOISE_CHARGE,
        )
    }

    /// Evaluates channel currents and operating state at forward-frame `[vgs, vds, vbs]`
    /// (`vds >= 0`).
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
        let vbseff = smooth_floor(vbs, self.vbsc);
        let vbseff = -smooth_floor(-vbseff, -(self.phi * 0.95));
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
        let t3 = above_dibl_floor(vbseff.scale(p.etab) + c(self.eta0));
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
        let weff = effective_width(c(self.weff) - (vgsteff.scale(p.dwg) + t9.scale(p.dwb)) * 2.0);
        let t0 = vgsteff.scale(p.prwg) + 1.0;
        let t2 = t0.recip() + t9.scale(p.prwb);
        let rds = smooth_positive(t2, D::number(0.01))
            .scale(self.rds0)
            .plus(self.rdswmin);

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
        let denominator = above_fifth(t5);
        let ueff = denominator.recip().scale(self.u0temp);

        // The saturation voltage.
        let wv_cox = weff.scale(self.vsattemp * self.coxe);
        let wv_cox_rds = wv_cox * rds;
        let esat = ueff.recip().scale(self.vsattemp * 2.0);
        let mut esat_l = esat.scale(leff);
        let excess = non_saturation_excess(p.a1, p.a2, vgsteff);
        let vgst2vtm = vgsteff.plus(vtm * 2.0);
        let vdsat = saturation_voltage(vgst2vtm, esat_l, abulk, wv_cox_rds, excess);

        // The effective drain-source voltage.
        let mut vdseff = effective_drain(vds, vdsat, p.delta);
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
        let t1 = excess * 2.0 + 1.0 + wv_cox_rds * abulk;
        let vasat = t0 / t1;

        // The linear current, with the inversion layer's thickness.
        let tcen = inversion_centroid(vgsteff.plus(self.vtfbphi2).divide(self.toxp * 2.0e8));
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
        let vascbe = substrate_early_voltage(diff_vds, (p.pscbe1, p.pscbe2), (leff, self.litl));
        let idsa = idl * (diff_vds / vadibl + 1.0) * (diff_vds / vadits + 1.0);
        let idsa = idsa * ((va / vasat).ln() / cclm + 1.0);

        // The impact ionization's substrate current.
        let isub = impact_ionization_factor(diff_vds, (p.alpha0, p.alpha1, p.beta0), leff)
            .map_or(D::number(0.0), |factor| factor * idsa * vdseff);

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
            beta,
            ids_over_vds: ids,
            nstar: (xdep.recip() * EPSSI)
                .plus(self.coxe + p.cit)
                .scale(vtm / Q),
        }
    }

    /// Evaluates total gate-induced drain and source leakage currents into the body across
    /// all fingers at `[vgs, vds, vbs]`.
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
