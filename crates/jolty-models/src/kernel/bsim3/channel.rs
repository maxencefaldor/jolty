//! The drain current of a BSIM3 transistor and the substrate current of its impact
//! ionization, as ngspice 47's `b3ld.c` computes them, oriented as an n-channel's with
//! the drain at or above the source.
//!
//! Part of jolty's BSIM3, derived from BSIM3v3.3.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2004 Regents of the University of California. All rights reserved. Used
//! under the BSIM terms of use, which the repository's `NOTICE` reproduces.

use super::Mobility;
use super::prepare::Core;
use crate::Real;
use crate::kernel::bsim::{
    D, EPSSI, EXP_THRESHOLD, MAX_EXP, MIN_EXP, Q, above_dibl_floor, above_fifth, above_half,
    above_tenth, effective_drain, effective_width, factor_above_tenth, impact_ionization_factor,
    non_saturation_excess, poly_depletion, reciprocal_above_tenth, saturation_voltage,
    smooth_floor, substrate_early_voltage,
};

/// Channel evaluation result at one bias in the forward frame (`vds >= 0`), carrying
/// values and derivatives with respect to frame gate-source, drain-source, and
/// bulk-source voltages.
#[derive(Debug, Clone, Copy)]
pub(super) struct Operating<R: Real> {
    /// The drain current, `Ids`.
    pub(super) ids: D<R>,
    /// The substrate current, from the drain to the bulk, `Isub`.
    pub(super) isub: D<R>,
    /// The threshold voltage, `Vth`.
    pub(super) vth: D<R>,
    pub(super) vdsat: D<R>,
    /// The gate-source voltage the poly gate's depletion leaves, `Vgs_eff`.
    pub(super) vgs_eff: D<R>,
    pub(super) vbseff: D<R>,
    /// The surface potential's root at `vbseff`.
    pub(super) sqrt_phis: D<R>,
    /// The subthreshold swing factor, `n`.
    pub(super) n: D<R>,
    pub(super) vgsteff: D<R>,
    pub(super) vdseff: D<R>,
    /// The bulk charge factor before the body bias's `keta` scales it, as the flicker
    /// noise reads it, and after, and `Abulk0` after.
    pub(super) abulk_unscaled: D<R>,
    pub(super) abulk: D<R>,
    pub(super) abulk0: D<R>,
    pub(super) ueff: D<R>,
    pub(super) rds: D<R>,
}

/// Evaluates `t · (1 + 2t)` at `t = e^x`, flooring `e^x` below `-EXP_THRESHOLD`.
fn falloff<R: Real>(x: D<R>) -> D<R> {
    let t = if x.value.value() > -EXP_THRESHOLD {
        x.exp()
    } else {
        D::number(MIN_EXP)
    };
    t * (t * 2.0 + 1.0)
}

impl<R: Real> Core<R> {
    /// Evaluates channel currents and operating state at frame gate-source, drain-source,
    /// and bulk-source voltages (`vds >= 0`).
    #[expect(
        clippy::too_many_lines,
        clippy::similar_names,
        reason = "one pass through b3ld.c's drain current, in its order, by its names"
    )]
    pub(super) fn evaluate(&self, [vgs, vds, vbs]: [D<R>; 3]) -> Operating<R> {
        let p = &self.binned;
        let leff = self.leff;
        // The effective bulk-source voltage, bounded below by `vbsc <= -3` V and clamped
        // above by `vbs`. By `smooth_floor`'s identity, the smooth curve lies strictly
        // below `vbs` if and only if `vbs > 0`, equals `0` at `vbs = 0`, and lies in
        // `(vbsc, 0)` strictly above `vbs` for every `vbs < 0`. Testing `vbs > 0`
        // directly avoids floating-point chatter of `vbseff < vbs` near zero while
        // keeping `b3ld.c`'s left slope `|vbsc| / (|vbsc| + 0.001)` at `vbs = 0`.
        let vbseff = if vbs.value.value() > 0.0 {
            vbs
        } else {
            smooth_floor(vbs, self.vbsc)
        };
        let sqrt_phis = if vbseff.value.value() > 0.0 {
            D::constant(self.phis3) / (vbseff * 0.5).plus(self.phi)
        } else {
            (-vbseff).plus(self.phi).sqrt()
        };
        let xdep = sqrt_phis.scale(self.xdep0 / self.sqrt_phi);

        // The threshold voltage.
        let t3 = xdep.sqrt();
        let v0 = self.vbi - self.phi;
        let lt1 = (t3 * above_half(vbseff.scale(p.dvt2))).scale(self.factor1);
        let ltw = (t3 * above_half(vbseff.scale(p.dvt2w))).scale(self.factor1);
        let theta0 = falloff(-(lt1.recip().scale(p.dvt1 * leff * 0.5)));
        let delta_vth = theta0.scale(p.dvt0 * v0);
        let narrow =
            falloff(-(ltw.recip().scale(p.dvt1w * self.weff * leff * 0.5))).scale(p.dvt0w * v0);
        let t1 = (vbseff.scale(p.kt2).plus(p.kt1 + p.kt1l / leff))
            .scale(self.temperature_rise)
            .plus(self.k1ox * ((p.nlx / leff + 1.0).sqrt() - 1.0) * self.sqrt_phi);
        let tmp2 = self.tox * self.phi / (self.weff + p.w0);
        let t3 = above_dibl_floor(vbseff.scale(p.etab).plus(p.eta0));
        let dibl = (t3 * vds).scale(self.theta0vb0);
        let vth = sqrt_phis.scale(self.k1ox) - vbseff.scale(self.k2ox) - delta_vth - narrow
            + (vbseff.scale(p.k3b).plus(p.k3)).scale(tmp2)
            + t1
            - dibl
            + D::constant(self.vth0 * self.sign - self.k1 * self.sqrt_phi);

        // The subthreshold swing factor.
        let tmp2 = xdep.recip().scale(p.nfactor * EPSSI);
        let tmp3 = (vbseff.scale(p.cdscb) + vds.scale(p.cdscd)).plus(p.cdsc);
        let n = above_half((tmp2 + tmp3 * theta0).plus(p.cit).divide(self.cox));

        // The poly gate's depletion.
        let vgs_eff = poly_depletion(vgs, self.ngate, self.vfb + self.phi, self.cox, Q);
        let vgst = vgs_eff - vth;

        // The effective gate overdrive, smooth through the threshold.
        let vtm = self.vtm;
        let t10 = n.scale(vtm * 2.0);
        let vgst_nvt = vgst / t10;
        let exp_arg = (-vgst).plus(p.voff * 2.0) / t10;
        let vgsteff = if vgst_nvt.value.value() > EXP_THRESHOLD {
            vgst
        } else if exp_arg.value.value() > EXP_THRESHOLD {
            let t0 = vgst.plus(-p.voff) / n.scale(vtm);
            t0.exp().scale(vtm * self.cdep0 / self.cox)
        } else {
            let t1 = t10 * (vgst_nvt.exp() + 1.0).ln();
            let t2 = (t10 * exp_arg.exp()).scale(self.cox / (vtm * self.cdep0)) + 1.0;
            t1 / t2
        };

        // The channel's effective width and the parasitic resistance.
        let t9 = sqrt_phis.plus(-self.sqrt_phi);
        let weff = effective_width(
            (vgsteff.scale(p.dwg) + t9.scale(p.dwb))
                .scale(R::constant(-2.0))
                .plus(self.weff),
        );
        let rds =
            factor_above_tenth(vgsteff.scale(p.prwg) + t9.scale(p.prwb), true).scale(self.rds0);

        // The bulk charge factor.
        let t1 = sqrt_phis.recip().scale(self.k1ox * 0.5);
        let t9 = xdep.scale(p.xj).sqrt();
        let t5 = D::constant(leff) / (t9 * 2.0).plus(leff);
        let t2 = t5.scale(p.a0).plus(p.b0 / (self.weff + p.b1));
        let t7 = t5 * t5 * t5;
        let abulk0 = t1 * t2 + 1.0;
        let abulk = abulk0 - t1 * t7.scale(p.ags * p.a0) * vgsteff;
        let abulk0 = above_tenth(abulk0);
        let abulk_unscaled = above_tenth(abulk);
        let keta = reciprocal_above_tenth(vbseff.scale(p.keta));
        let abulk = abulk_unscaled * keta;
        let abulk0 = abulk0 * keta;

        // The effective mobility.
        let tox = self.tox;
        let t5 = match self.mobmod {
            Mobility::One => {
                let t3 = (vgsteff + vth + vth).divide(tox);
                t3 * (vbseff.scale(self.uc).plus(self.ua) + t3.scale(self.ub))
            }
            Mobility::Two => {
                vgsteff.divide(tox)
                    * (vbseff.scale(self.uc).plus(self.ua) + vgsteff.divide(tox).scale(self.ub))
            }
            Mobility::Three => {
                let t3 = (vgsteff + vth + vth).divide(tox);
                t3 * (t3.scale(self.ub).plus(self.ua)) * (vbseff.scale(self.uc) + 1.0)
            }
        };
        let denominator = above_fifth(t5);
        let ueff = denominator.recip().scale(self.u0temp);

        // The saturation voltage.
        let wv_cox = weff.scale(self.vsattemp * self.cox);
        let wv_cox_rds = wv_cox * rds;
        let esat_l = ueff.recip().scale(self.vsattemp * 2.0 * leff);
        let excess = non_saturation_excess(p.a1, p.a2, vgsteff);
        let vgst2vtm = vgsteff.plus(vtm * 2.0);
        let vdsat = saturation_voltage(vgst2vtm, esat_l, abulk, wv_cox_rds, excess);

        // The effective drain-source voltage, smooth through saturation.
        let mut vdseff = effective_drain(vds, vdsat, p.delta);

        // The saturation's Early voltage.
        let tmp4 = -(abulk * vdsat / vgst2vtm * 0.5) + 1.0;
        let t0 = esat_l + vdsat + wv_cox_rds * vgsteff * tmp4 * 2.0;
        let t1 = excess * 2.0 + 1.0 + wv_cox_rds * abulk;
        let vasat = t0 / t1;
        if vdseff.value.value() > vds.value.value() {
            vdseff.value = vds.value;
        }
        let diff_vds = vds - vdseff;

        // The channel-length modulation's Early voltage.
        let vaclm = if p.pclm.value() > 0.0 && diff_vds.value.value() > 1.0e-10 {
            let t0 = abulk.scale(p.pclm * self.litl).recip();
            let t1 = (abulk + vgsteff / esat_l).scale(leff);
            t0 * t1 * diff_vds
        } else {
            D::number(MAX_EXP)
        };

        // The drain-induced barrier lowering's Early voltage.
        let vadibl = if self.theta_rout.value() > 0.0 {
            let t8 = abulk * vdsat;
            let t0 = vgst2vtm * t8;
            let t1 = vgst2vtm + t8;
            let vadibl = (vgst2vtm - t0 / t1).divide(self.theta_rout);
            vadibl * reciprocal_above_tenth(vbseff.scale(p.pdiblcb))
        } else {
            D::number(MAX_EXP)
        };

        // The Early voltage.
        let t9 = (vgsteff / esat_l).scale(p.pvag);
        let t0 = factor_above_tenth(t9, false);
        let va = vasat + t0 * (vaclm * vadibl / (vaclm + vadibl));

        // The substrate current's body effect on the output resistance.
        let vascbe = substrate_early_voltage(diff_vds, (p.pscbe1, p.pscbe2), (leff, self.litl));

        // The drain current.
        let beta = ueff * weff.scale(self.cox / leff);
        let fgche1 = vgsteff * (-(abulk * vdseff / vgst2vtm * 0.5) + 1.0);
        let fgche2 = (vdseff / esat_l) + 1.0;
        let gche = beta * fgche1 / fgche2;
        let idl = gche * vdseff / (gche * rds + 1.0);
        let idsa = idl * (diff_vds / va + 1.0);
        let ids = idsa * (diff_vds / vascbe + 1.0);

        // The substrate current.
        let isub = impact_ionization_factor(diff_vds, (p.alpha0, p.alpha1, p.beta0), leff)
            .map_or(D::number(0.0), |factor| factor * idsa);
        Operating {
            ids,
            isub,
            vth,
            vdsat,
            vgs_eff,
            vbseff,
            sqrt_phis,
            n,
            vgsteff,
            vdseff,
            abulk_unscaled,
            abulk,
            abulk0,
            ueff,
            rds,
        }
    }
}
