//! The channel of SPICE's level 3 MOSFET: a semi-empirical law of short channels.

use super::Conduction;
use crate::dual::Dual;

/// The coefficients of the depletion's lateral reach into a short channel, in SPICE's
/// polynomial of the depletion width over the junction depth.
const SHORT_CHANNEL: [f64; 3] = [0.063_135_3, 0.801_329_2, -0.011_107_77];

/// The level 3 channel, oriented as an n-channel's, as ngspice implements it: the threshold
/// falls with the drain voltage and rises with a narrow width, the gate field degrades the
/// mobility, carriers' velocity saturates, and the channel shortens beyond saturation.
#[derive(Debug, Clone, Copy)]
pub(super) struct Semiempirical {
    /// `kp · W / L`, with every parallel copy counted.
    pub(super) beta: f64,
    /// The effective channel length.
    pub(super) length: f64,
    pub(super) surface_potential: f64,
    pub(super) body_effect: f64,
    /// The threshold less the body effect at zero bias, `vbi`.
    pub(super) built_in: f64,
    /// How much the drain lowers the threshold, per volt.
    pub(super) static_feedback: f64,
    /// The narrow-width factor over the effective width.
    pub(super) narrow: f64,
    /// The gate field's degradation of the mobility, `θ`, per volt.
    pub(super) mobility_modulation: f64,
    /// `κ`, which scales the field that shortens the channel in saturation.
    pub(super) saturation_field_factor: f64,
    /// `2 · εsi / (q · nsub)`, zero without a substrate doping.
    pub(super) alpha: f64,
    /// The junction depth, zero to leave short channels' body effect out.
    pub(super) junction_depth: f64,
    pub(super) lateral_diffusion: f64,
    /// The carriers' maximum drift velocity, zero for none.
    pub(super) max_drift_velocity: f64,
    /// The surface mobility, in square meters per volt-second.
    pub(super) mobility: f64,
    /// The thermal voltage at the transistor's temperature.
    pub(super) thermal_voltage: f64,
    /// The fast surface states' capacitance relative to the oxide's, zero for none,
    /// which turns on weak inversion.
    pub(super) surface_states: f64,
}

impl Semiempirical {
    /// The drain current at the gate-source, drain-source and bulk-source voltages.
    pub(super) fn evaluate(&self, [vgs, vds, vbs]: [Dual<3>; 3]) -> Conduction {
        let forward = vds.value >= 0.0;
        let (vgs, vds, vbs) = if forward {
            (vgs, vds, vbs)
        } else {
            (vgs - vds, -vds, vbs - vds)
        };
        let potential = self.surface_potential;
        let (phibs, root) = if vbs.value <= 0.0 {
            let phibs = -vbs + potential;
            (phibs, phibs.sqrt())
        } else {
            let root = Dual::constant(potential.sqrt()) / (vbs / (2.0 * potential) + 1.0);
            (root * root, root)
        };
        let gamma = self.short_channel(root) * self.body_effect;
        let fbody = gamma * 0.25 / root + self.narrow;
        let charge = gamma * root + phibs * self.narrow;
        let vth = -vds * self.static_feedback + self.built_in + charge;
        let weak = (self.surface_states != 0.0)
            .then(|| charge / (phibs * 2.0) + (1.0 + self.surface_states));
        let von = weak.map_or(vth, |xn| vth + xn * self.thermal_voltage);
        let off = Conduction {
            current: Dual::constant(0.0),
            threshold: von,
            saturation: Dual::constant(0.0),
            forward,
        };
        if weak.is_none() && vgs.value <= von.value {
            return off;
        }
        let gate = if vgs.value >= von.value { vgs } else { von };
        let onfg = (gate - vth) * self.mobility_modulation + 1.0;
        let overdrive = (gate - vth) / (fbody + 1.0);
        let critical = (self.max_drift_velocity > 0.0)
            .then(|| onfg * (self.length * self.max_drift_velocity / self.mobility));
        let vdsat = critical.map_or(overdrive, |vdsc| {
            overdrive + vdsc - (overdrive * overdrive + vdsc * vdsc).sqrt()
        });
        let drain = if vds.value <= vdsat.value { vds } else { vdsat };
        let cdo = gate - vth - (fbody + 1.0) * drain * 0.5;
        let mut current = cdo * drain * self.beta / onfg;
        let fdrain = critical.map(|vdsc| (drain / vdsc + 1.0).recip());
        if let Some(fdrain) = fdrain {
            current = current * fdrain;
        }
        if let Some(shortening) = self.shortening(vds, vdsat, current, critical.zip(fdrain)) {
            current = current / (-(shortening / self.length) + 1.0);
        }
        if let Some(xn) = weak
            && vgs.value < von.value
        {
            current = current * ((vgs - von) / (xn * self.thermal_voltage)).exp();
        }
        Conduction {
            current: if forward { current } else { -current },
            threshold: von,
            saturation: vdsat,
            forward,
        }
    }

    /// The factor by which the source's and the drain's depletion share the channel's
    /// charge, with the square root `root` of the surface potential less the bulk-source
    /// voltage.
    fn short_channel(&self, root: Dual<3>) -> Dual<3> {
        if self.junction_depth == 0.0 || self.alpha == 0.0 {
            return Dual::constant(1.0);
        }
        let [c0, c1, c2] = SHORT_CHANNEL;
        let reach = root * (self.alpha.sqrt() / self.junction_depth);
        let spread = reach * reach * c2 + reach * c1 + c0;
        let diffusion = self.lateral_diffusion / self.junction_depth;
        let ratio = reach / (reach + 1.0);
        let cosine = (-(ratio * ratio) + 1.0).sqrt();
        -(((spread + diffusion) * cosine - diffusion) * (self.junction_depth / self.length)) + 1.0
    }

    /// How far the channel shortens at `vds`, with saturation voltage `vdsat`, the
    /// `current` so far, and the critical voltage and velocity-saturation factor of
    /// carriers' drift, if they saturate; `None` where it does not.
    fn shortening(
        &self,
        vds: Dual<3>,
        vdsat: Dual<3>,
        current: Dual<3>,
        drift: Option<(Dual<3>, Dual<3>)>,
    ) -> Option<Dual<3>> {
        let (kappa, alpha, length) = (self.saturation_field_factor, self.alpha, self.length);
        if alpha == 0.0 {
            return None;
        }
        let shortening = if vds.value <= vdsat.value {
            if drift.is_some() {
                return None;
            }
            let ratio = vds / vdsat;
            let square = ratio * ratio;
            (vdsat * (kappa * alpha / 8.0)).sqrt() * square * square
        } else if let Some((vdsc, fdrain)) = drift {
            let conductance = (current * (-fdrain + 1.0) / vdsc).max(1e-12);
            let field = current * kappa / (conductance * length);
            let half = field * (0.5 * alpha);
            (half * half + (vds - vdsat) * (kappa * alpha)).sqrt() - half
        } else {
            ((vds - vdsat + vdsat / 8.0) * (kappa * alpha)).sqrt()
        };
        // Punch-through: the shortening never reaches the whole channel.
        Some(if shortening.value > 0.5 * length {
            -(Dual::constant(length * length) / (shortening * 4.0)) + length
        } else {
            shortening
        })
    }
}
