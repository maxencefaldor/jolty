//! The charges of a BSIM3 transistor, as ngspice 47's `b3ld.c` computes them: the
//! intrinsic charges of `capmod` 0 to 3 under the gate, partitioned between the drain and
//! the source by `xpart`, the overlaps' charges, and the junctions' depletion charges.
//!
//! Part of jolty's BSIM3, derived from BSIM3v3.3.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2004 Regents of the University of California. All rights reserved. Used
//! under the BSIM terms of use, which the repository's `NOTICE` reproduces.

use super::channel::{Operating, effective_drain};
use super::prepare::{Core, Junction, Junctions, Overlaps};
use super::{Charges, D, DELTA, EPSSI, EXP_THRESHOLD, MAX_EXP, MIN_EXP};
use crate::Real;

/// The intrinsic charges on the gate, the bulk and the drain, in the frame where the drain
/// lies at or above the source, the source's their sum's opposite, and the inversion
/// charge the thermal noise of `noimod` 2 and 4 reads.
#[derive(Debug, Clone, Copy)]
pub(super) struct Intrinsic<R: Real> {
    pub(super) gate: D<R>,
    pub(super) bulk: D<R>,
    pub(super) drain: D<R>,
    pub(super) inversion: f64,
}

/// How `xpart` shares the channel's charge in saturation: the drain takes none, 40 % or
/// half.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Partition {
    /// `xpart` above 0.5: 0/100.
    None,
    /// `xpart` below 0.5: 40/60.
    Forty,
    /// `xpart` of 0.5: 50/50.
    Half,
}

impl Partition {
    fn from_xpart(xpart: f64) -> Self {
        if xpart > 0.5 {
            Self::None
        } else if xpart < 0.5 {
            Self::Forty
        } else {
            Self::Half
        }
    }
}

/// `x0 − ½ · (v + √(v² ± 4δ · x0))`, the smooth minimum of `b3ld.c`'s effective
/// flat-band voltage, with `v = x0 − x − δ`, the sign of the root's term that of `x0`.
pub(crate) fn flat_band<R: Real>(vfb: R, v3: D<R>) -> D<R> {
    let t0 = if vfb.value() <= 0.0 {
        (v3 * v3 - D::constant(vfb * (4.0 * DELTA))).sqrt()
    } else {
        (v3 * v3 + D::constant(vfb * (4.0 * DELTA))).sqrt()
    };
    (-((v3 + t0) * 0.5)).plus(vfb)
}

/// The depletion charge under the gate, `CoxWL · k1ox · (√(k1ox²/4 + v) − k1ox/2)`,
/// continued linearly below `v = 0`, with `capacitance` the gate's oxide capacitance.
pub(crate) fn depletion<R: Real>(k1ox: R, v: D<R>, capacitance: D<R>) -> D<R> {
    let t0 = k1ox * 0.5;
    let t1 = if k1ox.value() == 0.0 {
        D::number(0.0)
    } else if v.value.value() < 0.0 {
        v.divide(k1ox).plus(t0)
    } else {
        (v.plus(t0 * t0)).sqrt()
    };
    capacitance * t1.plus(-t0).scale(k1ox)
}

impl<R: Real> Core<R> {
    /// The intrinsic charges at the drain-source and bulk-source voltages `vds` and `vbs`
    /// of the frame where the drain lies at or above the source, where the channel is
    /// `channel`.
    pub(super) fn intrinsic(&self, vds: D<R>, vbs: D<R>, channel: &Operating<R>) -> Intrinsic<R> {
        let zero = Intrinsic {
            gate: D::number(0.0),
            bulk: D::number(0.0),
            drain: D::number(0.0),
            inversion: 0.0,
        };
        if self.xpart < 0.0 {
            return zero;
        }
        let partition = Partition::from_xpart(self.xpart);
        match self.capmod {
            Charges::Zero => self.capmod_zero(vds, vbs, channel, partition),
            Charges::One => self.capmod_one(&self.cv(vds, channel, partition)),
            Charges::Two => self.capmod_two(&self.cv(vds, channel, partition)),
            Charges::Three => self.capmod_three(&self.cv(vds, channel, partition)),
        }
    }

    /// What the charges of `capmod` 1 to 3 read at the drain-source voltage `vds` where
    /// the channel is `channel`: the bulk-source voltage of the charges and their gate
    /// overdrive, which `noff` and `voffcv` shift.
    fn cv<'a>(&self, vds: D<R>, channel: &'a Operating<R>, partition: Partition) -> Cv<'a, R> {
        // For `vbseff > 0`, `phi - phis = phi - phi^2 / (phi + vbseff) =
        // phi * vbseff / (phi + vbseff)`, eliminating `phi - phi` cancellation near zero.
        // At `vbseff == 0`, keep `channel.vbseff` and its left slope `|vbsc| / (|vbsc| +
        // 0.001)`, matching ngspice 47's `b3ld.c` lines 1585 to 1592.
        let vbseff = if channel.vbseff.value.value() <= 0.0 {
            channel.vbseff
        } else {
            channel.vbseff.scale(self.phi) / channel.vbseff.plus(self.phi)
        };
        let noff = channel.n.scale(self.binned.noff);
        let t0 = noff.scale(self.vtm);
        let voffcv = self.binned.voffcv;
        let vgst = channel.vgs_eff - channel.vth;
        let vgst_nvt = vgst.plus(-voffcv) / t0;
        let vgsteff = if vgst_nvt.value.value() > EXP_THRESHOLD {
            vgst.plus(-voffcv)
        } else if vgst_nvt.value.value() < -EXP_THRESHOLD {
            t0 * (1.0 + MIN_EXP).ln()
        } else {
            t0 * (vgst_nvt.exp() + 1.0).ln()
        };
        Cv {
            vds,
            vbseff,
            vgsteff,
            channel,
            partition,
        }
    }

    /// The oxide capacitance of the channel's area, `Cox · WeffCV · LeffCV`.
    fn cox_wl(&self) -> R {
        self.cox * self.weff_cv * self.leff_cv
    }

    /// `capmod` 0: the long-channel charges of the original BSIM3, piecewise in the
    /// regions of operation.
    fn capmod_zero(
        &self,
        vds: D<R>,
        vbs: D<R>,
        channel: &Operating<R>,
        partition: Partition,
    ) -> Intrinsic<R> {
        // In `capmod` 0, `VbseffCV` is `vbs` for `vbs <= 0` and
        // `phi - phis = phi * vbs / (phi + vbs)` for `vbs > 0`. Both one-sided
        // derivatives at `vbs = 0` equal `1`, so `VbseffCV` is C¹ smooth through zero,
        // matching ngspice 47's `b3ld.c` lines 1267 to 1274. Testing `channel.vbseff < 0`
        // previously sent `vbs = 0` into `phi - channel.phis`, which evaluated to
        // `channel.vbseff` and inherited its left slope `|vbsc| / (|vbsc| + 0.001)`
        // instead of `1`.
        let vbseff = if vbs.value.value() <= 0.0 {
            vbs
        } else {
            vbs.scale(self.phi) / vbs.plus(self.phi)
        };
        let vfb = self.binned.vfbcv;
        let phi = self.phi;
        let vgs_eff = channel.vgs_eff;
        let vth = channel.sqrt_phis.scale(self.k1ox).plus(vfb + phi);
        let vgst = vgs_eff - vth;
        let cox_wl = self.cox_wl();
        let arg1 = (vgs_eff - vbseff).plus(-vfb);
        let opposite = |gate: D<R>| Intrinsic {
            gate,
            bulk: -gate,
            drain: D::number(0.0),
            inversion: 0.0,
        };
        if arg1.value.value() <= 0.0 {
            return opposite(arg1.scale(cox_wl));
        }
        if vgst.value.value() <= 0.0 {
            let t1 = self.k1ox * 0.5;
            let t2 = arg1.plus(t1 * t1).sqrt();
            return opposite(t2.plus(-t1).scale(cox_wl * self.k1ox));
        }
        let abulk_cv = channel.abulk0.scale(self.abulk_cv_factor);
        let vdsat = vgst / abulk_cv;
        let surface = vgs_eff.plus(-vfb).plus(-phi);
        let (gate, bulk, drain) = if vds.value.value() >= vdsat.value.value() {
            let gate = (surface - vdsat / 3.0).scale(cox_wl);
            let t2 = vgst.scale(cox_wl * (-2.0 / 3.0));
            let drain = match partition {
                Partition::None => D::number(0.0),
                Partition::Forty => t2 * 0.4,
                Partition::Half => t2 * 0.5,
            };
            (gate, -(gate + t2), drain)
        } else {
            let alphaz = vgst / vdsat;
            let t1 = vdsat * 2.0 - vds;
            let t2 = vds / (t1 * 3.0);
            let t3 = t2 * vds;
            let t4 = alphaz.scale(cox_wl * 0.25);
            let gate = (surface - (vds - t3) * 0.5).scale(cox_wl);
            match partition {
                Partition::None => {
                    let t7 = vds * 2.0 - t1 - t3 * 3.0;
                    let t8 = t3 - t1 - vds * 2.0;
                    let drain = t4 * t7;
                    (gate, -(gate + drain + t4 * t8), drain)
                }
                Partition::Forty => {
                    let t6 = vdsat * vdsat * 8.0 - vdsat * vds * 6.0 + vds * vds * 1.2;
                    let t8 = t2 / t1;
                    let drain = t4 * (vds - t1 - t8 * t6);
                    (gate, -(gate - t4 * (t1 + t3) * 2.0), drain)
                }
                Partition::Half => {
                    let drain = -(t4 * (t1 + t3));
                    (gate, -(gate + drain + drain), drain)
                }
            }
        };
        Intrinsic {
            gate,
            bulk,
            drain,
            inversion: -(gate + bulk).value.value(),
        }
    }

    /// `capmod` 1: charges continuous in every region but piecewise across saturation.
    fn capmod_one(&self, cv: &Cv<'_, R>) -> Intrinsic<R> {
        let Cv {
            vds,
            vbseff,
            vgsteff,
            channel,
            partition,
        } = *cv;
        let cox_wl = self.cox_wl();
        let arg1 = (channel.vgs_eff - vbseff - vgsteff).plus(-self.vfbzb);
        let mut gate = if arg1.value.value() <= 0.0 {
            arg1.scale(cox_wl)
        } else {
            let t0 = self.k1ox * 0.5;
            arg1.plus(t0 * t0)
                .sqrt()
                .plus(-t0)
                .scale(cox_wl * self.k1ox)
        };
        let mut bulk = -gate;
        let abulk_cv = channel.abulk0.scale(self.abulk_cv_factor);
        let vdsat_cv = vgsteff / abulk_cv;
        let source = if vdsat_cv.value.value() < vds.value.value() {
            gate = gate + (vgsteff - vdsat_cv / 3.0).scale(cox_wl);
            bulk = bulk + (vdsat_cv - vgsteff).scale(cox_wl / 3.0);
            let share = match partition {
                Partition::None => -2.0 / 3.0,
                Partition::Forty => -0.4,
                Partition::Half => -1.0 / 3.0,
            };
            vgsteff.scale(cox_wl * share)
        } else {
            let t0 = abulk_cv * vds;
            let t1 = (vgsteff - t0 * 0.5 + 1.0e-20) * 12.0;
            let t3 = t0 * (vds / t1);
            let inversion_gate = (vgsteff - vds * 0.5 + t3).scale(cox_wl);
            let inversion_bulk = ((-abulk_cv + 1.0) * (vds * 0.5 - t3)).scale(cox_wl);
            gate = gate + inversion_gate;
            bulk = bulk + inversion_bulk;
            self.partitioned(partition, (vgsteff, t0, t1), (cox_wl, gate + bulk))
        };
        Intrinsic {
            gate,
            bulk,
            drain: -(gate + bulk + source),
            inversion: -(gate + bulk).value.value(),
        }
    }

    /// The source's share of the inversion charge below saturation, of `capmod` 1 and 2:
    /// with the gate overdrive, `Abulk · Vds` and `12 · (Vgsteff − Abulk · Vds / 2)`, the
    /// channel's oxide capacitance and the inversion charges on the gate and the bulk.
    fn partitioned(
        &self,
        partition: Partition,
        (vgsteff, t0, t1): (D<R>, D<R>, D<R>),
        (cox_wl, inversion): (R, D<R>),
    ) -> D<R> {
        let _ = self;
        match partition {
            Partition::None => {
                let t1 = t1 * 2.0;
                -((vgsteff * 0.5 + t0 * 0.25 - t0 * t0 / t1).scale(cox_wl))
            }
            Partition::Forty => {
                let t1 = t1 / 12.0;
                let t2 = (t1 * t1).recip().scale(cox_wl * 0.5);
                let t3 = vgsteff * (t0 * t0 * (2.0 / 3.0) + vgsteff * (vgsteff - t0 * (4.0 / 3.0)))
                    - t0 * t0 * t0 * (2.0 / 15.0);
                -(t2 * t3)
            }
            Partition::Half => -(inversion * 0.5),
        }
    }

    /// `capmod` 2: smooth charges in every region.
    fn capmod_two(&self, cv: &Cv<'_, R>) -> Intrinsic<R> {
        let Cv {
            vds,
            vbseff,
            vgsteff,
            channel,
            partition,
        } = *cv;
        let cox_wl = self.cox_wl();
        let vfb = self.vfbzb;
        let vgs_eff = channel.vgs_eff;
        let vfbeff = flat_band(vfb, (vbseff - vgs_eff).plus(vfb - DELTA));
        let qac0 = vfbeff.plus(-vfb).scale(cox_wl);
        let qsub0 = depletion(
            self.k1ox,
            vgs_eff - vfbeff - vbseff - vgsteff,
            D::constant(cox_wl),
        );
        let abulk_cv = channel.abulk0.scale(self.abulk_cv_factor);
        let vdsat_cv = vgsteff / abulk_cv;
        let vdseff_cv = effective_drain(vds, vdsat_cv, R::constant(DELTA));
        let t0 = abulk_cv * vdseff_cv;
        let t1 = (vgsteff - t0 * 0.5 + 1.0e-20) * 12.0;
        let t3 = t0 * (vdseff_cv / t1);
        let inversion = -((vgsteff - t0 * 0.5 + abulk_cv * t3).scale(cox_wl));
        let gate = (vgsteff - vdseff_cv * 0.5 + t3).scale(cox_wl);
        let bulk = ((-abulk_cv + 1.0) * (vdseff_cv * 0.5 - t3)).scale(cox_wl);
        let source = self.partitioned(partition, (vgsteff, t0, t1), (cox_wl, gate + bulk));
        let gate = gate + qac0 + qsub0;
        let bulk = bulk - (qac0 + qsub0);
        Intrinsic {
            gate,
            bulk,
            drain: -(gate + bulk + source),
            inversion: inversion.value.value(),
        }
    }

    /// `capmod` 3: the charge-thickness model, with the inversion layer's finite
    /// thickness and the gate bias's effect on the surface potential.
    fn capmod_three(&self, cv: &Cv<'_, R>) -> Intrinsic<R> {
        let Cv {
            vds,
            vbseff,
            vgsteff,
            channel,
            partition,
        } = *cv;
        let cox_wl = self.cox_wl();
        let vfbzb = self.vfbzb;
        let vgs_eff = channel.vgs_eff;
        let vfbeff = flat_band(vfbzb, (vbseff - vgs_eff).plus(vfbzb - DELTA));
        let cox = self.cox;
        let tox = self.tox * 1.0e8;
        let ldeb = self.ldeb;
        // The accumulation and depletion charge's centroid.
        let t0 = (vgs_eff - vbseff).plus(-vfbzb).divide(tox);
        let tmp = t0.scale(self.acde);
        let tcen = if tmp.value.value() > -EXP_THRESHOLD && tmp.value.value() < EXP_THRESHOLD {
            tmp.exp().scale(ldeb)
        } else if tmp.value.value() <= -EXP_THRESHOLD {
            D::constant(ldeb * MIN_EXP)
        } else {
            D::constant(ldeb * MAX_EXP)
        };
        let link = self.tox * 1.0e-3;
        let v3 = (-tcen).plus(ldeb - link);
        let v4 = (v3 * v3).plus(link * ldeb * 4.0).sqrt();
        let tcen = (-((v3 + v4) * 0.5)).plus(ldeb);
        let effective = |tcen: D<R>| {
            let ccen = D::number(EPSSI) / tcen;
            let coxeff = ccen * (ccen.plus(cox)).recip().scale(cox);
            coxeff.divide(cox).scale(cox_wl)
        };
        let cox_wl_cen = effective(tcen);
        let qac0 = cox_wl_cen * vfbeff.plus(-vfbzb);
        let qsub0 = depletion(self.k1ox, vgs_eff - vfbeff - vbseff - vgsteff, cox_wl_cen);

        // The gate bias's effect on the surface potential.
        let vtm = self.vtm;
        let (denominator, t0) = if self.k1ox.value() <= 0.0 {
            (self.binned.moin * vtm * 0.25, self.sqrt_phi * 0.5)
        } else {
            (
                self.binned.moin * vtm * self.k1ox * self.k1ox,
                self.k1ox * self.sqrt_phi,
            )
        };
        let t1 = vgsteff.plus(t0 * 2.0);
        let delta_phi = ((t1 * vgsteff).divide(denominator) + 1.0).ln().scale(vtm);
        let t0 = vgsteff - delta_phi - 0.001;
        let t1 = (t0 * t0 + vgsteff * 0.004).sqrt();
        let vgdp = (t0 + t1) * 0.5;

        // The inversion charge's centroid.
        let t3 = (channel.vth.plus(-(vfbzb + self.phi))) * 4.0;
        let tox = tox * 2.0;
        let t0 = if t3.value.value() >= 0.0 {
            (vgsteff + t3).divide(tox)
        } else {
            (vgsteff + 1.0e-20).divide(tox)
        };
        let tmp = (t0.ln() * 0.7).exp();
        let tcen = D::number(1.9e-9) / (tmp + 1.0);
        let cox_wl_cen = effective(tcen);

        let abulk_cv = channel.abulk0.scale(self.abulk_cv_factor);
        let vdsat_cv = vgdp / abulk_cv;
        let vdseff_cv = effective_drain(vds, vdsat_cv, R::constant(DELTA));
        let t0 = abulk_cv * vdseff_cv;
        let t1 = vgdp;
        let t2 = (t1 - t0 * 0.5 + 1.0e-20) * 12.0;
        let t3 = t0 / t2;
        let inversion = cox_wl_cen * (t1 - t0 * (-t3 + 0.5));
        let bulk = cox_wl_cen * (-abulk_cv + 1.0) * (vdseff_cv * 0.5 - t0 * vdseff_cv / t2);
        let source = match partition {
            Partition::None => -(cox_wl_cen * (t1 * 0.5 + t0 * 0.25 - t0 * t0 * 0.5 / t2)),
            Partition::Forty => {
                let t2 = t2 / 12.0;
                let t3 = cox_wl_cen * 0.5 / (t2 * t2);
                let t4 = t1 * (t0 * t0 * (2.0 / 3.0) + t1 * (t1 - t0 * (4.0 / 3.0)))
                    - t0 * t0 * t0 * (2.0 / 15.0);
                -(t3 * t4)
            }
            Partition::Half => -(inversion * 0.5),
        };
        let gate = inversion + qac0 + qsub0 - bulk;
        let bulk = bulk - (qac0 + qsub0);
        Intrinsic {
            gate,
            bulk,
            drain: -(gate + bulk + source),
            inversion: -inversion.value.value(),
        }
    }
}

/// The voltages and the overdrive the charges of `capmod` 1 to 3 read.
#[derive(Clone, Copy)]
struct Cv<'a, R: Real> {
    vds: D<R>,
    /// The bulk-source voltage of the charges, `VbseffCV`.
    vbseff: D<R>,
    /// The gate overdrive of the charges, with `noff` and `voffcv`.
    vgsteff: D<R>,
    channel: &'a Operating<R>,
    partition: Partition,
}

impl<R: Real> Overlaps<R> {
    /// The gate-source and gate-drain overlap charges at `vgs` and `vgd`, of `capmod`, and
    /// the gate-bulk one at `vgb`.
    pub(super) fn charges(&self, capmod: Charges, [vgs, vgd, vgb]: [D<R>; 3]) -> [D<R>; 3] {
        let ckappa = self.ckappa;
        let side = |v: D<R>, cgo: R, cgl: R| match capmod {
            Charges::Zero => v.scale(cgo),
            Charges::One => {
                if v.value.value() < 0.0 {
                    let t1 = (-(v.divide(ckappa) * 4.0) + 1.0).sqrt();
                    v.scale(cgo) - (t1 - 1.0).scale(cgl * ckappa * 0.5)
                } else {
                    v.scale(cgl + cgo)
                }
            }
            Charges::Two | Charges::Three => {
                let t0 = v + DELTA;
                let t1 = (t0 * t0 + 4.0 * DELTA).sqrt();
                let t2 = (t0 - t1) * 0.5;
                let t4 = (-(t2.divide(ckappa) * 4.0) + 1.0).sqrt();
                v.scale(cgo + cgl) - (t2 + (t4 - 1.0).scale(ckappa * 0.5)).scale(cgl)
            }
        };
        [
            side(vgs, self.cgso, self.cgsl),
            side(vgd, self.cgdo, self.cgdl),
            vgb.scale(self.cgbo),
        ]
    }
}

impl<R: Real> Junctions<R> {
    /// The depletion charge of `junction` at its voltage `v`, the bulk's over the drain's
    /// or the source's: the power law of each part in reverse bias, and in forward bias a
    /// capacitance growing linearly from its zero-bias value with the slope it has there.
    pub(super) fn charge(&self, junction: &Junction<R>, v: D<R>) -> D<R> {
        let mut charge = D::number(0.0);
        if v.value.value() < 0.0 {
            for ((&capacitance, &potential), &grading) in junction
                .capacitances
                .iter()
                .zip(&self.potentials)
                .zip(&self.gradings)
            {
                if capacitance.value() > 0.0 {
                    let arg = -v.divide(potential) + 1.0;
                    #[expect(clippy::float_cmp, reason = "b3ld.c's own test for a square root")]
                    let root = grading.value() == 0.5;
                    let sarg = if root {
                        arg.sqrt().recip()
                    } else {
                        (arg.ln().scale(-grading)).exp()
                    };
                    charge = charge
                        + (-(arg * sarg) + 1.0).scale(potential * capacitance / (-grading + 1.0));
                }
            }
            return charge;
        }
        let mut total = R::constant(0.0);
        let mut slope = R::constant(0.0);
        for ((&capacitance, &potential), &grading) in junction
            .capacitances
            .iter()
            .zip(&self.potentials)
            .zip(&self.gradings)
        {
            total += capacitance;
            slope += capacitance * grading / potential;
        }
        v * (v.scale(slope * 0.5)).plus(total)
    }
}
