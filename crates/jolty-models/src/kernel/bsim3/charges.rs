//! The charges of a BSIM3 transistor, as ngspice 47's `b3ld.c` computes them: the
//! intrinsic charges of `capmod` 0 to 3 under the gate, partitioned between the drain and
//! the source by `xpart`, the overlaps' charges, and the junctions' depletion charges.
//!
//! Part of jolty's BSIM3, derived from BSIM3v3.3.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2004 Regents of the University of California. All rights reserved. Used
//! under the BSIM terms of use, which the repository's `NOTICE` reproduces.

use super::Charges;
use super::channel::Operating;
use super::prepare::{Core, Junction, Junctions, Overlaps};
use crate::Real;
use crate::kernel::bsim::{
    D, DELTA, Partition, accumulation_centroid, charge_thickness_charges, cv_overdrive, depletion,
    effective_drain, effective_oxide_capacitance, flat_band, inversion_centroid, smooth_overlap,
    source_partition_charge, surface_potential_overdrive, three_part_depletion,
};

/// Intrinsic gate, bulk, and drain charges in the forward frame (`vds >= 0`), where the
/// source charge negates their sum, plus the inversion charge for `noimod` 2 and 4 thermal
/// noise.
#[derive(Debug, Clone, Copy)]
pub(super) struct Intrinsic<R: Real> {
    pub(super) gate: D<R>,
    pub(super) bulk: D<R>,
    pub(super) drain: D<R>,
    pub(super) inversion: R,
}

impl<R: Real> Core<R> {
    /// Evaluates intrinsic terminal charges at forward-frame `vds` and `vbs` for operating
    /// state `channel`.
    pub(super) fn intrinsic(&self, vds: D<R>, vbs: D<R>, channel: &Operating<R>) -> Intrinsic<R> {
        let zero = Intrinsic {
            gate: D::number(0.0),
            bulk: D::number(0.0),
            drain: D::number(0.0),
            inversion: R::constant(0.0),
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

    /// Computes the effective bulk-source voltage and `noff`/`voffcv`-shifted gate overdrive
    /// used by `capmod` 1 to 3 at `vds` and `channel`.
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
        let vgsteff = cv_overdrive(
            channel.vgs_eff,
            channel.vth,
            channel.n,
            (self.binned.noff, self.binned.voffcv, self.vtm),
        );
        Cv {
            vds,
            vbseff,
            vgsteff,
            channel,
            partition,
        }
    }

    /// Returns the total channel oxide capacitance `Cox · WeffCV · LeffCV`.
    fn cox_wl(&self) -> R {
        self.cox * self.weff_cv * self.leff_cv
    }

    /// Evaluates `capmod` 0 long-channel charges, defined piecewise across operating regions.
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
            inversion: R::constant(0.0),
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
            inversion: -(gate + bulk).value,
        }
    }

    /// Evaluates `capmod` 1 charges, continuous across threshold and piecewise across
    /// saturation.
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
            source_partition_charge(
                partition,
                (vgsteff, t0, t1),
                D::constant(cox_wl),
                gate + bulk,
            )
        };
        Intrinsic {
            gate,
            bulk,
            drain: -(gate + bulk + source),
            inversion: -(gate + bulk).value,
        }
    }

    /// Evaluates `capmod` 2 smooth charges across all operating regions.
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
        let source = source_partition_charge(
            partition,
            (vgsteff, t0, t1),
            D::constant(cox_wl),
            gate + bulk,
        );
        let gate = gate + qac0 + qsub0;
        let bulk = bulk - (qac0 + qsub0);
        Intrinsic {
            gate,
            bulk,
            drain: -(gate + bulk + source),
            inversion: inversion.value,
        }
    }

    /// Evaluates `capmod` 3 charge-thickness charges, modeling finite inversion-layer
    /// thickness and gate-bias effects on surface potential.
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
        // The accumulation and depletion charge's centroid.
        let tcen = accumulation_centroid(
            (vgs_eff - vbseff).plus(-vfbzb),
            self.tox,
            self.acde,
            self.ldeb,
        );
        let cox_wl_cen = effective_oxide_capacitance(tcen, self.cox, self.cox, cox_wl);
        let qac0 = cox_wl_cen * vfbeff.plus(-vfbzb);
        let qsub0 = depletion(self.k1ox, vgs_eff - vfbeff - vbseff - vgsteff, cox_wl_cen);

        // The gate bias's effect on the surface potential.
        let vgdp = surface_potential_overdrive(
            vgsteff,
            self.k1ox,
            self.sqrt_phi,
            self.binned.moin,
            self.vtm,
        );

        // The inversion charge's centroid.
        let t3 = channel.vth.plus(-(vfbzb + self.phi)) * 4.0;
        let tox = self.tox * 2.0e8;
        let field = if t3.value.value() >= 0.0 {
            (vgsteff + t3).divide(tox)
        } else {
            (vgsteff + 1.0e-20).divide(tox)
        };
        let cox_wl_cen =
            effective_oxide_capacitance(inversion_centroid(field), self.cox, self.cox, cox_wl);

        let abulk_cv = channel.abulk0.scale(self.abulk_cv_factor);
        let [gate, bulk, drain, inversion] =
            charge_thickness_charges(vds, vgdp, abulk_cv, cox_wl_cen, (qac0, qsub0), partition);
        Intrinsic {
            gate,
            bulk,
            drain,
            inversion: -inversion.value,
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
    /// Evaluates gate-source, gate-drain, and gate-bulk overlap charges for `capmod` at
    /// `vgs`, `vgd`, and `vgb`.
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
            Charges::Two | Charges::Three => smooth_overlap(v, cgo, cgl, ckappa),
        };
        [
            side(vgs, self.cgso, self.cgsl),
            side(vgd, self.cgdo, self.cgdl),
            vgb.scale(self.cgbo),
        ]
    }
}

impl<R: Real> Junctions<R> {
    /// Evaluates the depletion charge of `junction` at bulk-over-terminal voltage `v`.
    /// Reverse bias follows the power law of each junction component, and forward bias
    /// continues linearly from zero-bias capacitance and slope.
    pub(super) fn charge(&self, junction: &Junction<R>, v: D<R>) -> D<R> {
        three_part_depletion(junction.capacitances, self.potentials, self.gradings, v)
    }
}
