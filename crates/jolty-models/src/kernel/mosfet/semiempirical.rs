//! Semi-empirical short-channel law for the SPICE Level 3 MOSFET.
//!
//! Derived from ngspice 47 `src/spicelib/devices/mos3/`: Copyright 1985 - 2018, Regents of
//! the University of California and others. Used under the ngspice Modified BSD license,
//! reproduced in the repository `NOTICE` file.

use super::{Conduction, Region};
use crate::Real;
use crate::kernel::dual::Dual;

/// SPICE polynomial coefficients modeling lateral depletion reach into a short channel as a
/// function of depletion width over junction depth.
const SHORT_CHANNEL: [f64; 3] = [0.063_135_3, 0.801_329_2, -0.011_107_77];

/// Level 3 MOSFET channel parameters in n-channel sign convention, matching ngspice.
///
/// Captures drain-induced barrier lowering, narrow-width threshold rise, gate-field mobility
/// degradation, carrier velocity saturation, and channel-length modulation past saturation.
#[derive(Debug, Clone, Copy)]
pub(super) struct Semiempirical<R: Real> {
    /// Scaled transconductance gain `kp · W / L`, including parallel multiplier.
    pub(super) beta: R,
    /// Effective channel length.
    pub(super) length: R,
    pub(super) surface_potential: R,
    pub(super) body_effect: R,
    /// Zero-bias built-in threshold component `vbi` (threshold minus body effect).
    pub(super) built_in: R,
    /// Drain-induced threshold lowering coefficient, per volt.
    pub(super) static_feedback: R,
    /// Narrow-width threshold factor divided by effective channel width.
    pub(super) narrow: R,
    /// Gate-field mobility degradation coefficient `θ`, per volt.
    pub(super) mobility_modulation: R,
    /// Saturation field factor `κ` scaling channel-length shortening in saturation.
    pub(super) saturation_field_factor: R,
    /// Depletion factor `2 · εsi / (q · nsub)`, or zero when substrate doping is absent.
    pub(super) alpha: R,
    /// Junction depth, or zero to omit short-channel body-effect sharing.
    pub(super) junction_depth: R,
    pub(super) lateral_diffusion: R,
    /// Maximum carrier drift velocity, or zero when velocity saturation is disabled.
    pub(super) max_drift_velocity: R,
    /// Surface carrier mobility, in square meters per volt-second.
    pub(super) mobility: R,
    /// Thermal voltage at operating temperature.
    pub(super) thermal_voltage: R,
    /// Fast surface-state capacitance normalized to oxide capacitance; nonzero enables weak
    /// inversion.
    pub(super) surface_states: R,
}

impl<R: Real> Semiempirical<R> {
    /// Evaluates Level 3 drain current, threshold, saturation voltage, and operating region
    /// at `[vgs, vds, vbs]`.
    pub(super) fn evaluate(&self, [vgs, vds, vbs]: [Dual<R, 3>; 3]) -> Conduction<R> {
        let forward = vds.value.value() >= 0.0;
        let (vgs, vds, vbs) = if forward {
            (vgs, vds, vbs)
        } else {
            (vgs - vds, -vds, vbs - vds)
        };
        let potential = self.surface_potential;
        let (phibs, root) = if vbs.value.value() <= 0.0 {
            let phibs = (-vbs).plus(potential);
            (phibs, phibs.sqrt())
        } else {
            let root = Dual::constant(potential.sqrt()) / (vbs.divide(potential * 2.0) + 1.0);
            (root * root, root)
        };
        let gamma = self.short_channel(root).scale(self.body_effect);
        let fbody = (gamma * 0.25 / root).plus(self.narrow);
        let charge = gamma * root + phibs.scale(self.narrow);
        let vth = (-vds).scale(self.static_feedback).plus(self.built_in) + charge;
        let weak = (self.surface_states.value() != 0.0)
            .then(|| (charge / (phibs * 2.0)).plus(self.surface_states + 1.0));
        let von = weak.map_or(vth, |xn| vth + xn.scale(self.thermal_voltage));
        let above = (vgs - von).value.value();
        if weak.is_none() && above <= 0.0 {
            return Conduction {
                current: Dual::number(0.0),
                threshold: von,
                saturation: Dual::number(0.0),
                forward,
                region: Region::Off,
                margins: [above, vds.value.value()],
            };
        }
        let gate = if above >= 0.0 { vgs } else { von };
        let onfg = (gate - vth).scale(self.mobility_modulation) + 1.0;
        let overdrive = (gate - vth) / (fbody + 1.0);
        let critical = (self.max_drift_velocity.value() > 0.0)
            .then(|| onfg.scale(self.length * self.max_drift_velocity / self.mobility));
        let vdsat = critical.map_or(overdrive, |vdsc| {
            overdrive + vdsc - (overdrive * overdrive + vdsc * vdsc).sqrt()
        });
        let margin = (vds - vdsat).value.value();
        let region = if above < 0.0 {
            Region::WeakInversion
        } else if margin <= 0.0 {
            Region::Linear
        } else {
            Region::Saturated
        };
        let drain = if margin <= 0.0 { vds } else { vdsat };
        let cdo = gate - vth - (fbody + 1.0) * drain * 0.5;
        let mut current = (cdo * drain).scale(self.beta) / onfg;
        let fdrain = critical.map(|vdsc| (drain / vdsc + 1.0).recip());
        if let Some(fdrain) = fdrain {
            current = current * fdrain;
        }
        if let Some(shortening) = self.shortening(vds, vdsat, current, critical.zip(fdrain)) {
            current = current / (-(shortening.divide(self.length)) + 1.0);
        }
        if let Some(xn) = weak
            && region == Region::WeakInversion
        {
            current = current * ((vgs - von) / xn.scale(self.thermal_voltage)).exp();
        }
        Conduction {
            current: if forward { current } else { -current },
            threshold: von,
            saturation: vdsat,
            forward,
            region,
            margins: [above, margin],
        }
    }

    /// Computes the short-channel charge-sharing factor from `root = √(φ − vbs)`.
    fn short_channel(&self, root: Dual<R, 3>) -> Dual<R, 3> {
        if self.junction_depth.value() == 0.0 || self.alpha.value() == 0.0 {
            return Dual::number(1.0);
        }
        let [c0, c1, c2] = SHORT_CHANNEL;
        let reach = root.scale(self.alpha.sqrt() / self.junction_depth);
        let spread = reach * reach * c2 + reach * c1 + c0;
        let diffusion = self.lateral_diffusion / self.junction_depth;
        let ratio = reach / (reach + 1.0);
        let cosine = (-(ratio * ratio) + 1.0).sqrt();
        -((spread.plus(diffusion) * cosine)
            .plus(-diffusion)
            .scale(self.junction_depth / self.length))
            + 1.0
    }

    /// Computes channel-length shortening at `vds`, saturation voltage `vdsat`, preliminary
    /// `current`, and optional velocity-saturation `drift` parameters, returning `None` when
    /// shortening is inactive.
    fn shortening(
        &self,
        vds: Dual<R, 3>,
        vdsat: Dual<R, 3>,
        current: Dual<R, 3>,
        drift: Option<(Dual<R, 3>, Dual<R, 3>)>,
    ) -> Option<Dual<R, 3>> {
        let (kappa, alpha, length) = (self.saturation_field_factor, self.alpha, self.length);
        // Either factor zero makes the shortening zero at every bias, its derivatives
        // too, as ngspice computes it.
        if alpha.value() == 0.0 || kappa.value() == 0.0 {
            return None;
        }
        let shortening = if vds.value.value() <= vdsat.value.value() {
            if drift.is_some() {
                return None;
            }
            let ratio = vds / vdsat;
            let square = ratio * ratio;
            vdsat.scale(kappa * alpha / 8.0).sqrt() * square * square
        } else if let Some((vdsc, fdrain)) = drift {
            let conductance = (current * (-fdrain + 1.0) / vdsc).max(1e-12);
            let field = current.scale(kappa) / conductance.scale(length);
            let half = field.scale(alpha * 0.5);
            (half * half + (vds - vdsat).scale(kappa * alpha)).sqrt() - half
        } else {
            (vds - vdsat + vdsat / 8.0).scale(kappa * alpha).sqrt()
        };
        // Punch-through: the shortening never reaches the whole channel.
        Some(if shortening.value.value() > 0.5 * length.value() {
            (-(Dual::constant(length * length) / (shortening * 4.0))).plus(length)
        } else {
            shortening
        })
    }
}
