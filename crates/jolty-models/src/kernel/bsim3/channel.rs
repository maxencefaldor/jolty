//! The drain current of a BSIM3 transistor and the substrate current of its impact
//! ionization, as ngspice 47's `b3ld.c` computes them, oriented as an n-channel's with
//! the drain at or above the source.
//!
//! Part of jolty's BSIM3, derived from BSIM3v3.3.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2004 Regents of the University of California. All rights reserved. Used
//! under the BSIM terms of use, which the repository's `NOTICE` reproduces.

use super::prepare::Core;
use super::{D, EPSSI, EXP_THRESHOLD, MAX_EXP, MIN_EXP, Mobility, Q};
use crate::Real;

/// What the channel yields at one bias, in the frame where the drain lies at or above the
/// source, each with its derivatives with respect to that frame's gate-source,
/// drain-source and bulk-source voltages.
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
    /// The surface potential and its root at `vbseff`.
    pub(super) phis: D<R>,
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

/// The clamp of `b3ld.c` that keeps a factor `1 + x` from approaching zero: `1 + x` down
/// to `x = −0.9`, then `(17 + 20x) / (0.8 + x)`'s reciprocal, its first form's value and
/// slope continued.
pub(crate) fn reciprocal_above_tenth<R: Real>(x: D<R>) -> D<R> {
    if x.value.value() >= -0.9 {
        (x + 1.0).recip()
    } else {
        (x * 20.0 + 17.0) / (x + 0.8)
    }
}

/// `1 + x`, continued below `x = −0.9` by `(0.8 + x) / (17 + 20x)`, as `b3ld.c` keeps
/// such a factor positive.
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

/// `1 + x`, continued below `x = −0.5` by `(1 + 3x) / (3 + 8x)`.
pub(crate) fn above_half<R: Real>(x: D<R>) -> D<R> {
    if x.value.value() >= -0.5 {
        x + 1.0
    } else {
        (x * 3.0 + 1.0) / (x * 8.0 + 3.0)
    }
}

/// `t · (1 + 2t)` at `t = e^x`, with `e^x` at its floor below `−EXP_THRESHOLD`.
fn falloff<R: Real>(x: D<R>) -> D<R> {
    let t = if x.value.value() > -EXP_THRESHOLD {
        x.exp()
    } else {
        D::number(MIN_EXP)
    };
    t * (t * 2.0 + 1.0)
}

impl<R: Real> Core<R> {
    /// The channel at the gate-source, drain-source and bulk-source voltages, the drain at
    /// or above the source.
    #[expect(
        clippy::too_many_lines,
        clippy::similar_names,
        reason = "one pass through b3ld.c's drain current, in its order, by its names"
    )]
    pub(super) fn evaluate(&self, [vgs, vds, vbs]: [D<R>; 3]) -> Operating<R> {
        let p = &self.binned;
        let leff = self.leff;
        // The effective bulk-source voltage, bounded below by `vbsc`.
        let t0 = (vbs - 0.001).plus(-self.vbsc);
        let t1 = (t0 * t0 - D::constant(self.vbsc * 0.004)).sqrt();
        let mut vbseff = ((t0 + t1) * 0.5).plus(self.vbsc);
        if vbseff.value.value() < vbs.value.value() {
            vbseff = vbs;
        }
        let (phis, sqrt_phis) = if vbseff.value.value() > 0.0 {
            let t0 = D::constant(self.phi) / vbseff.plus(self.phi);
            (
                t0.scale(self.phi),
                D::constant(self.phis3) / (vbseff * 0.5).plus(self.phi),
            )
        } else {
            let phis = (-vbseff).plus(self.phi);
            (phis, phis.sqrt())
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
        let mut t3 = vbseff.scale(p.etab).plus(p.eta0);
        if t3.value.value() < 1.0e-4 {
            // Keeps the drain-induced barrier lowering from turning negative.
            t3 = (-t3 + 2.0e-4) / (-(t3 * 2.0e4) + 3.0);
        }
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
        let flat = self.vfb + self.phi;
        let ngate = self.ngate.value();
        let vgs_eff = if ngate > 1.0e18 && ngate < 1.0e25 && vgs.value.value() > flat.value() {
            let t1 = self.ngate * (1.0e6 * Q * EPSSI) / (self.cox * self.cox);
            let t4 = ((vgs.plus(-flat) * 2.0).divide(t1) + 1.0).sqrt();
            let t2 = (t4 - 1.0).scale(t1);
            let t3 = (t2 * t2 * 0.5).divide(t1);
            let t7 = -t3 + (1.12 - 0.05);
            let t6 = (t7 * t7 + 0.224).sqrt();
            let t5 = -((t7 + t6) * 0.5) + 1.12;
            vgs - t5
        } else {
            vgs
        };
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
        let mut weff = (vgsteff.scale(p.dwg) + t9.scale(p.dwb))
            .scale(R::constant(-2.0))
            .plus(self.weff);
        if weff.value.value() < 2.0e-8 {
            weff = (-weff + 4.0e-8) * 2.0e-8 / (-(weff * 2.0) + 6.0e-8);
        }
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
        let denominator = if t5.value.value() >= -0.8 {
            t5 + 1.0
        } else {
            (t5 + 0.6) / (t5 * 10.0 + 7.0)
        };
        let ueff = denominator.recip().scale(self.u0temp);

        // The saturation voltage.
        let wv_cox = weff.scale(self.vsattemp * self.cox);
        let wv_cox_rds = wv_cox * rds;
        let esat_l = ueff.recip().scale(self.vsattemp * 2.0 * leff);
        let a1 = p.a1.value();
        let lambda = if a1 == 0.0 {
            D::constant(p.a2)
        } else if a1 > 0.0 {
            let t0 = -p.a2 + 1.0;
            let t1 = (-vgsteff.scale(p.a1)).plus(t0 - 0.0001);
            let t2 = (t1 * t1).plus(t0 * 0.0004).sqrt();
            (-((t1 + t2) * 0.5)).plus(p.a2 + t0)
        } else {
            let t1 = vgsteff.scale(p.a1).plus(p.a2 - 0.0001);
            let t2 = (t1 * t1).plus(p.a2 * 0.0004).sqrt();
            (t1 + t2) * 0.5
        };
        let vgst2vtm = vgsteff.plus(vtm * 2.0);
        #[expect(clippy::float_cmp, reason = "b3ld.c's own test of the exact default")]
        let unclamped = rds.value.value() == 0.0 && lambda.value.value() == 1.0;
        let vdsat = if unclamped {
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

        // The effective drain-source voltage, smooth through saturation.
        let delta = p.delta;
        let t1 = vdsat - vds - D::constant(delta);
        let t2 = (t1 * t1 + vdsat.scale(delta * 4.0)).sqrt();
        let mut vdseff = vdsat - (t1 + t2) * 0.5;
        if vds.value.value() == 0.0 {
            // As `b3ld.c` does, with none of the rounding the smoothing leaves at zero.
            vdseff.value = vdseff.value.with_value(0.0);
            let [gate, _, bulk] = &mut vdseff.gradient;
            *gate = gate.with_value(0.0);
            *bulk = bulk.with_value(0.0);
        }

        // The saturation's Early voltage.
        let tmp4 = -(abulk * vdsat / vgst2vtm * 0.5) + 1.0;
        let t0 = esat_l + vdsat + wv_cox_rds * vgsteff * tmp4 * 2.0;
        let t1 = lambda.recip() * 2.0 - 1.0 + wv_cox_rds * abulk;
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
        let vascbe = if p.pscbe2.value() > 0.0 {
            if diff_vds.value.value() > p.pscbe1.value() * self.litl.value() / EXP_THRESHOLD {
                let t0 = diff_vds.recip().scale(p.pscbe1 * self.litl);
                t0.exp().scale(leff / p.pscbe2)
            } else {
                D::constant(leff / p.pscbe2 * MAX_EXP)
            }
        } else {
            D::number(MAX_EXP)
        };

        // The drain current.
        let beta = ueff * weff.scale(self.cox / leff);
        let fgche1 = vgsteff * (-(abulk * vdseff / vgst2vtm * 0.5) + 1.0);
        let fgche2 = (vdseff / esat_l) + 1.0;
        let gche = beta * fgche1 / fgche2;
        let idl = gche * vdseff / (gche * rds + 1.0);
        let idsa = idl * (diff_vds / va + 1.0);
        let ids = idsa * (diff_vds / vascbe + 1.0);

        // The substrate current.
        let tmp = p.alpha0 + p.alpha1 * leff;
        let isub = if tmp.value() <= 0.0 || p.beta0.value() <= 0.0 {
            D::number(0.0)
        } else {
            let t2 = tmp / leff;
            let t1 = if diff_vds.value.value() > p.beta0.value() / EXP_THRESHOLD {
                (diff_vds * (-diff_vds.recip().scale(p.beta0)).exp()).scale(t2)
            } else {
                diff_vds.scale(t2 * MIN_EXP)
            };
            t1 * idsa
        };
        Operating {
            ids,
            isub,
            vth,
            vdsat,
            vgs_eff,
            vbseff,
            phis,
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
