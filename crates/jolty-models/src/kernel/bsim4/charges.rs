//! A BSIM4 transistor's charges: the channel's of `capmod` 2, the charge-thickness model,
//! the overlaps' with their bias dependence, and the junctions' depletion charges, as
//! ngspice 47's `b4v5ld.c` computes them.
//!
//! Part of jolty's BSIM4, derived from BSIM4.5.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2005 Regents of the University of California. All rights reserved. Used
//! under the BSIM4 terms of use, which the repository's `NOTICE` reproduces.

use super::channel::Operating;
use super::prepare::{Core, Overlaps};
use super::{D, J};
use crate::Real;
use crate::kernel::bsim3::{
    DELTA, EPSSI, EXP_THRESHOLD, MAX_EXP, MIN_EXP, depletion, flat_band, smooth_positive,
};

/// The channel's charges at one bias, in the frame whose drain lies at or above its
/// source: on the gate, on the body and on the frame's drain.
pub(super) struct Intrinsic<R: Real> {
    pub(super) gate: D<R>,
    pub(super) bulk: D<R>,
    pub(super) drain: D<R>,
}

/// The depletion charge of one junction's three parts, the bottom, the isolation
/// sidewall and the gate sidewall, at its zero-bias capacitances, built-in potentials and
/// grading coefficients.
#[derive(Debug, Clone, Copy)]
pub(super) struct Depletion<R: Real> {
    pub(super) capacitances: [R; 3],
    pub(super) potentials: [R; 3],
    pub(super) gradings: [R; 3],
}

impl<R: Real> Depletion<R> {
    /// The charge on the body node against the diffusion at the junction's voltage `v`:
    /// each part's depletion charge in reverse, and its linear continuation forward.
    pub(super) fn charge(&self, v: J<R>) -> J<R> {
        let value = v.value.value();
        if value < 0.0 {
            self.capacitances
                .iter()
                .zip(self.potentials)
                .zip(self.gradings)
                .fold(
                    J::number(0.0),
                    |sum, ((&capacitance, potential), grading)| {
                        if capacitance.value() <= 0.0 {
                            return sum;
                        }
                        let arg = 1.0 - v.divide(potential);
                        #[expect(
                            clippy::float_cmp,
                            reason = "b4v5ld.c takes the square root at exactly 0.5"
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
            let [bottom, sidewall, gate] = self.capacitances;
            let total = bottom + sidewall + gate;
            let slope = self
                .capacitances
                .iter()
                .zip(self.potentials)
                .zip(self.gradings)
                .fold(
                    R::constant(0.0),
                    |sum, ((&capacitance, potential), grading)| {
                        sum + capacitance * grading / potential
                    },
                );
            v * (v.scale(slope * 0.5) + J::constant(total))
        }
    }
}

impl<R: Real> Core<R> {
    /// The channel's charges of `capmod` 2 at the frame's drain-source voltage `vds`,
    /// where the channel is `operating`, its drain's share `xpart`.
    #[expect(
        clippy::similar_names,
        reason = "b4v5ld.c's names of the charges of capmod 2"
    )]
    pub(super) fn intrinsic(&self, vds: D<R>, operating: &Operating<R>) -> Intrinsic<R> {
        let p = &self.binned;
        let c = D::constant;
        let vtm = self.vtm;
        let vbseff_cv = operating.vbseff;
        let cox_wl = self.coxe * self.weff_cv * self.leff_cv * self.fingers;
        let noff = operating.n.scale(p.noff);
        let t0 = noff.scale(vtm);
        let vgst = operating.vgs_eff - operating.vth;
        let vgst_nvt = vgst.plus(-p.voffcv) / t0;
        let vgsteff = if vgst_nvt.value.value() > EXP_THRESHOLD {
            vgst.plus(-p.voffcv)
        } else if vgst_nvt.value.value() < -EXP_THRESHOLD {
            t0 * MIN_EXP.ln_1p()
        } else {
            t0 * vgst_nvt.exp().ln_1p()
        };
        let vgs_eff = operating.vgs_eff;
        let vfbzb = self.vfbzb;

        // Accumulation.
        let vfbeff = flat_band(vfbzb, c(vfbzb) - vgs_eff + vbseff_cv - DELTA);
        let cox = self.coxp;
        let tox = self.toxp * 1.0e8;
        let t0 = (vgs_eff - vbseff_cv).plus(-vfbzb).divide(tox);
        let tmp = t0.scale(self.acde);
        let ldeb = self.ldeb;
        let tcen = if -EXP_THRESHOLD < tmp.value.value() && tmp.value.value() < EXP_THRESHOLD {
            tmp.exp().scale(ldeb)
        } else if tmp.value.value() <= -EXP_THRESHOLD {
            c(ldeb * MIN_EXP)
        } else {
            c(ldeb * MAX_EXP)
        };
        let link = self.toxp * 1.0e-3;
        let v3 = c(ldeb - link) - tcen;
        let v4 = (v3 * v3 + c(link * ldeb * 4.0)).sqrt();
        let tcen = c(ldeb) - (v3 + v4) * 0.5;
        let ccen = tcen.recip() * EPSSI;
        let coxeff = (ccen + c(cox)).recip().scale(cox) * ccen;
        let cox_wl_cen = coxeff.scale(cox_wl / self.coxe);
        let qac0 = cox_wl_cen * vfbeff.plus(-vfbzb);

        // Depletion.
        let k1ox = self.k1ox;
        let qsub0 = depletion(k1ox, vgs_eff - vfbeff - vbseff_cv - vgsteff, cox_wl_cen);

        // Inversion: the surface potential's rise with the gate.
        let (denominator, t0) = if k1ox.value() <= 0.0 {
            (p.moin * vtm * 0.25, self.sqrt_phi * 0.5)
        } else {
            (p.moin * vtm * k1ox * k1ox, k1ox * self.sqrt_phi)
        };
        let t1 = vgsteff.plus(t0 * 2.0);
        let delta_phi = (t1 * vgsteff).divide(denominator).ln_1p().scale(vtm);
        let t0 = vgsteff - delta_phi - 0.001;
        let vg_dp = smooth_positive(t0, vgsteff * 0.004);
        let t0 = vgsteff.plus(self.vtfbphi2).divide(tox * 2.0);
        let tmp = (t0.ln() * 0.7).exp();
        let tcen = (tmp + 1.0).recip() * 1.9e-9;
        let ccen = tcen.recip() * EPSSI;
        let coxeff = (ccen + c(cox)).recip().scale(cox) * ccen;
        let cox_wl_cen = coxeff.scale(cox_wl / self.coxe);
        let abulk_cv = operating.abulk0.scale(self.abulk_cv_factor);
        let vdsat_cv = vg_dp / abulk_cv;
        let t0 = vdsat_cv - vds - DELTA;
        let t1 = (t0 * t0 + vdsat_cv * (4.0 * DELTA)).sqrt();
        let mut vdseff_cv = if t0.value.value() >= 0.0 {
            vdsat_cv - (t0 + t1) * 0.5
        } else {
            let t3 = (t1 - t0).recip() * (DELTA + DELTA);
            vdsat_cv * (1.0 - t3)
        };
        if vds.value.value() == 0.0 {
            let [gate, drain, bulk] = vdseff_cv.gradient;
            vdseff_cv = D {
                value: vdseff_cv.value.with_value(0.0),
                gradient: [gate.with_value(0.0), drain, bulk.with_value(0.0)],
            };
        }
        let t0 = abulk_cv * vdseff_cv;
        let t1 = vg_dp;
        let t2 = (t1 - t0 * 0.5 + 1.0e-20) * 12.0;
        let t3 = t0 / t2;
        let qgate = cox_wl_cen * (t1 - t0 * (0.5 - t3));
        let t7 = 1.0 - abulk_cv;
        let qbulk = cox_wl_cen * t7 * (vdseff_cv * 0.5 - t0 * vdseff_cv / t2);
        let qsrc = if self.xpart > 0.5 {
            -(cox_wl_cen * (t1 * 0.5 + t0 * 0.25 - t0 * t0 * 0.5 / t2))
        } else if self.xpart < 0.5 {
            let t2 = t2 * (1.0 / 12.0);
            let t3 = cox_wl_cen * 0.5 / (t2 * t2);
            let t4 = t1 * (t0 * t0 * (2.0 / 3.0) + t1 * (t1 - t0 * (4.0 / 3.0)))
                - t0 * t0 * t0 * (2.0 / 15.0);
            -(t3 * t4)
        } else {
            -(qgate * 0.5)
        };
        let qgate = qgate + qac0 + qsub0 - qbulk;
        let qbulk = qbulk - (qac0 + qsub0);
        Intrinsic {
            gate: qgate,
            bulk: qbulk,
            drain: -(qgate + qbulk + qsrc),
        }
    }
}

impl<R: Real> Overlaps<R> {
    /// The overlaps' charges on the gate against the source, the drain and the body at
    /// the gate-source, gate-drain and gate-body voltages `voltages`, each with the bias
    /// dependence of `capmod` 2.
    pub(super) fn charges(&self, [vgs, vgd, vgb]: [D<R>; 3]) -> [D<R>; 3] {
        let overlap = |v: D<R>, fixed: R, lightly: R, kappa: R| {
            let t0 = v + DELTA;
            let t1 = (t0 * t0 + 4.0 * DELTA).sqrt();
            let t2 = (t0 - t1) * 0.5;
            let t4 = (1.0 - t2.divide(kappa) * 4.0).sqrt();
            (v.scale(fixed + lightly) - (t2 + (t4 - 1.0).scale(kappa * 0.5)).scale(lightly))
                .scale(self.fingers)
        };
        [
            overlap(vgs, self.cgso, self.cgsl, self.ckappas),
            overlap(vgd, self.cgdo, self.cgdl, self.ckappad),
            vgb.scale(self.cgbo),
        ]
    }
}
