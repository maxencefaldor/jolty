//! A BSIM4 transistor's charges: the channel's of `capmod` 2, the charge-thickness model,
//! the overlaps' with their bias dependence, and the junctions' depletion charges, as
//! ngspice 47's `b4v5ld.c` computes them.
//!
//! Part of jolty's BSIM4, derived from BSIM4.5.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2005 Regents of the University of California. All rights reserved. Used
//! under the BSIM4 terms of use, which the repository's `NOTICE` reproduces.

use super::J;
use super::channel::Operating;
use super::prepare::{Core, Overlaps};
use crate::Real;
use crate::kernel::bsim::{
    D, DELTA, Partition, accumulation_centroid, charge_thickness_charges, cv_overdrive, depletion,
    effective_oxide_capacitance, flat_band, inversion_centroid, smooth_overlap,
    surface_potential_overdrive, three_part_depletion,
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
    /// Evaluates depletion charge on the body against the diffusion at junction voltage `v`,
    /// using power laws in reverse bias and linear continuation in forward bias.
    pub(super) fn charge(&self, v: J<R>) -> J<R> {
        three_part_depletion(self.capacitances, self.potentials, self.gradings, v)
    }
}

impl<R: Real> Core<R> {
    /// Evaluates `capmod` 2 intrinsic channel charges at forward-frame `vds` and
    /// `operating`, partitioned by `xpart`.
    pub(super) fn intrinsic(&self, vds: D<R>, operating: &Operating<R>) -> Intrinsic<R> {
        let p = &self.binned;
        let bulk_bias = operating.vbseff;
        let channel_oxide = self.coxe * self.weff_cv * self.leff_cv * self.fingers;
        let overdrive = cv_overdrive(
            operating.vgs_eff,
            operating.vth,
            operating.n,
            (p.noff, p.voffcv, self.vtm),
        );
        let gate_source = operating.vgs_eff;
        let zero_flat_band = self.vfbzb;

        // Accumulation and depletion.
        let effective_flat_band = flat_band(
            zero_flat_band,
            D::constant(zero_flat_band) - gate_source + bulk_bias - DELTA,
        );
        let centroid = accumulation_centroid(
            (gate_source - bulk_bias).plus(-zero_flat_band),
            self.toxp,
            self.acde,
            self.ldeb,
        );
        let effective_oxide =
            effective_oxide_capacitance(centroid, self.coxp, self.coxe, channel_oxide);
        let accumulation = effective_oxide * effective_flat_band.plus(-zero_flat_band);
        let depletion_charge = depletion(
            self.k1ox,
            gate_source - effective_flat_band - bulk_bias - overdrive,
            effective_oxide,
        );

        // Inversion: the surface potential's rise with the gate.
        let surface_overdrive =
            surface_potential_overdrive(overdrive, self.k1ox, self.sqrt_phi, p.moin, self.vtm);
        let field = overdrive.plus(self.vtfbphi2).divide(self.toxp * 2.0e8);
        let effective_oxide = effective_oxide_capacitance(
            inversion_centroid(field),
            self.coxp,
            self.coxe,
            channel_oxide,
        );
        let abulk_cv = operating.abulk0.scale(self.abulk_cv_factor);
        let [gate, bulk, drain, _] = charge_thickness_charges(
            vds,
            surface_overdrive,
            abulk_cv,
            effective_oxide,
            (accumulation, depletion_charge),
            Partition::from_xpart(self.xpart),
        );
        Intrinsic { gate, bulk, drain }
    }
}

impl<R: Real> Overlaps<R> {
    /// Evaluates `capmod` 2 bias-dependent gate-source, gate-drain, and gate-body overlap
    /// charges at `[gate_source, gate_drain, gate_bulk]`.
    pub(super) fn charges(&self, [gate_source, gate_drain, gate_bulk]: [D<R>; 3]) -> [D<R>; 3] {
        let side = |voltage: D<R>, fixed: R, lightly: R, kappa: R| {
            smooth_overlap(voltage, fixed, lightly, kappa).scale(self.fingers)
        };
        [
            side(gate_source, self.cgso, self.cgsl, self.ckappas),
            side(gate_drain, self.cgdo, self.cgdl, self.ckappad),
            gate_bulk.scale(self.cgbo),
        ]
    }
}
