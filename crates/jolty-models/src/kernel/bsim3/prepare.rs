//! A BSIM3 transistor's numbers at its temperature and size, as ngspice 47's `b3temp.c`
//! prepares them, checked as its `b3check.c` checks them.
//!
//! Part of jolty's BSIM3, derived from BSIM3v3.3.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2004 Regents of the University of California. All rights reserved. Used
//! under the BSIM terms of use, which the repository's `NOTICE` reproduces.

use super::parameters::{Binned, Bsim3Parameters};
use super::{
    Charges, EPSOX, EPSSI, EXP_THRESHOLD, KBOQ, MIN_EXP, Mobility, NoiseModel, Q, by_channel,
};
use crate::kernel::parts::series_value;
use crate::{ParameterError, Real};

/// The intrinsic carrier concentration of silicon at 300.15 K, per cubic centimeter, as
/// `b3temp.c` takes it.
const INTRINSIC: f64 = 1.45e10;

/// The numbers the drain current and the charges of one transistor read, its channel's
/// size and temperature folded in. Concentrations are per cubic centimeter, as the
/// equations of `b3ld.c` take them.
#[derive(Debug, Clone, Copy)]
pub(super) struct Core<R: Real> {
    pub(super) mobmod: Mobility,
    pub(super) capmod: Charges,
    pub(super) xpart: f64,
    /// The channel's sign, 1 for an n-channel and −1 for a p-channel.
    pub(super) sign: f64,
    pub(super) tox: R,
    /// The oxide capacitance per area.
    pub(super) cox: R,
    /// `√(εsi / εox · tox)`.
    pub(super) factor1: R,
    /// The thermal voltage `k·T / q` with BSIM3's `k / q`.
    pub(super) vtm: R,
    /// `T / Tnom − 1`.
    pub(super) temperature_rise: R,
    pub(super) leff: R,
    pub(super) weff: R,
    pub(super) leff_cv: R,
    pub(super) weff_cv: R,
    pub(super) vbsc: R,
    pub(super) phi: R,
    pub(super) sqrt_phi: R,
    /// `phi^1.5`.
    pub(super) phis3: R,
    pub(super) xdep0: R,
    pub(super) vbi: R,
    pub(super) cdep0: R,
    pub(super) litl: R,
    pub(super) ldeb: R,
    pub(super) acde: R,
    pub(super) k1: R,
    pub(super) k1ox: R,
    pub(super) k2ox: R,
    pub(super) theta0vb0: R,
    pub(super) theta_rout: R,
    pub(super) abulk_cv_factor: R,
    pub(super) u0temp: R,
    pub(super) vsattemp: R,
    pub(super) rds0: R,
    pub(super) ua: R,
    pub(super) ub: R,
    pub(super) uc: R,
    /// The threshold at zero bias of a long, wide channel, signed as the channel conducts,
    /// `delvto` added.
    pub(super) vth0: R,
    pub(super) vfb: R,
    pub(super) vfbzb: R,
    /// The poly gate's doping, per cubic centimeter.
    pub(super) ngate: R,
    pub(super) binned: Binned<R>,
}

/// One junction between the bulk and the drain or the source.
#[derive(Debug, Clone, Copy)]
pub(super) struct Junction<R: Real> {
    pub(super) saturation_current: R,
    /// The voltage above which the current continues linearly, with the current
    /// `Is · e^(vjm / Nvtm)` there, where `ijth` sets one.
    pub(super) linear: Option<(R, R)>,
    /// The bottom, the field-oxide sidewall's and the gate sidewall's zero-bias
    /// capacitances.
    pub(super) capacitances: [R; 3],
}

/// The diodes between the bulk and the drain and the source, and their charges.
#[derive(Debug, Clone, Copy)]
pub(super) struct Junctions<R: Real> {
    /// `Nvtm`, the emission coefficient times the thermal voltage.
    pub(super) slope: R,
    pub(super) drain: Junction<R>,
    pub(super) source: Junction<R>,
    /// The bottom, the field-oxide sidewall's and the gate sidewall's built-in potentials
    /// and grading coefficients.
    pub(super) potentials: [R; 3],
    pub(super) gradings: [R; 3],
}

/// The overlaps between the gate and the source, the drain and the bulk.
#[derive(Debug, Clone, Copy)]
pub(super) struct Overlaps<R: Real> {
    /// `(cgso + cf) · WeffCV` and `(cgdo + cf) · WeffCV`.
    pub(super) cgso: R,
    pub(super) cgdo: R,
    /// `cgbo · LeffCV`.
    pub(super) cgbo: R,
    /// `cgsl · WeffCV` and `cgdl · WeffCV`.
    pub(super) cgsl: R,
    pub(super) cgdl: R,
    pub(super) ckappa: R,
}

/// The numbers of the noise models.
#[derive(Debug, Clone, Copy)]
pub(super) struct Noise {
    pub(super) model: NoiseModel,
    pub(super) noia: f64,
    pub(super) noib: f64,
    pub(super) noic: f64,
    pub(super) em: f64,
    pub(super) ef: f64,
    pub(super) af: f64,
    pub(super) kf: f64,
    /// The effective length less twice `lintnoi`.
    pub(super) length: f64,
}

/// A transistor's numbers at its temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedBsim3<R: Real> {
    pub(super) core: Core<R>,
    pub(super) junctions: Junctions<R>,
    pub(super) overlaps: Overlaps<R>,
    pub(super) noise: Noise,
    /// The drain and source resistances' stamped values: each conductance, or each
    /// resistance where its current is an unknown, of the parallel copies together.
    pub(super) series: [R; 2],
    pub(super) multiplier: R,
    pub(super) temperature: f64,
}

/// Fails with `message` when `failed` holds, as `b3check.c` stops on a fatal error.
pub(crate) fn check(failed: bool, message: impl FnOnce() -> String) -> Result<(), ParameterError> {
    if failed {
        Err(ParameterError(message()))
    } else {
        Ok(())
    }
}

/// The reciprocal-free length or width lost, `int + l / L^ln + w / W^wn + wl / (L^ln · W^wn)`.
pub(crate) fn lost<R: Real>(
    base: R,
    [by_length, by_width, by_area]: [R; 3],
    [length, width]: [R; 2],
) -> R {
    base + by_length / length + by_width / width + by_area / (length * width)
}

impl<R: Real> Bsim3Parameters<R> {
    /// The transistor's numbers at `temperature`, in kelvins.
    #[expect(
        clippy::too_many_lines,
        clippy::similar_names,
        reason = "one pass through b3temp.c, in its order, by its names"
    )]
    pub(super) fn prepared(&self, temperature: R) -> Result<PreparedBsim3<R>, ParameterError> {
        if temperature.value() <= 0.0 {
            return Err(ParameterError(format!(
                "a BSIM3 transistor's temperature lies above 0 K, not at {:?} K",
                temperature.value()
            )));
        }
        let tnom = ParameterError::positive("nominal_temperature", self.nominal_temperature)?;
        let multiplier = ParameterError::positive("multiplier", self.multiplier)?;
        let tox = self.tox;
        check(tox.value() <= 0.0, || {
            format!("tox must be positive, not {:?}", tox.value())
        })?;
        let toxm = self.toxm.unwrap_or(tox);
        check(toxm.value() <= 0.0, || {
            format!("toxm must be positive, not {:?}", toxm.value())
        })?;
        let cox = R::constant(EPSOX) / tox;
        let sign = self.channel.sign();
        let ratio = temperature / tnom;
        let rise = ratio - 1.0;
        let factor1 = (tox * (EPSSI / EPSOX)).sqrt();
        let vtm0 = tnom * KBOQ;
        let eg0 = -(tnom * tnom * 7.02e-4 / (tnom + 1108.0)) + 1.16;
        let reference = tnom / 300.15;
        let ni =
            reference * reference.sqrt() * INTRINSIC * (-(eg0 / (vtm0 * 2.0)) + 21.556_598_1).exp();
        let vtm = temperature * KBOQ;

        let geometry = self.geometry()?;
        let [leff, weff, leff_cv, weff_cv] = geometry;
        let inverse = [leff.recip(), weff.recip(), (leff * weff).recip()];
        let mut binned = self.binned(inverse);
        let abulk_cv_factor = (binned.clc / leff_cv).pow(binned.cle) + 1.0;
        let ua = binned.ua + binned.ua1 * rise;
        let ub = binned.ub + binned.ub1 * rise;
        let uc = binned.uc + binned.uc1 * rise;
        let u0temp = binned.u0 * ratio.pow(binned.ute);
        let vsattemp = binned.vsat - binned.at * rise;
        let mut rds0 = (binned.rdsw + binned.prt * rise) / (weff * 1e6).pow(binned.wr);
        self.checked(&binned, [leff, weff], [u0temp, vsattemp])?;
        let (mut cgso, mut cgdo, mut cgbo) = self.overlap_defaults(cox);
        if self.check_parameters {
            // ngspice's `paramchk = 1` clamps what it warns of.
            if binned.a2.value() < 0.01 {
                binned.a2 = R::constant(0.01);
            } else if binned.a2.value() > 1.0 {
                binned.a2 = R::constant(1.0);
                binned.a1 = R::constant(0.0);
            }
            if binned.rdsw.value() < 0.0 {
                binned.rdsw = R::constant(0.0);
                rds0 = R::constant(0.0);
            } else if rds0.value() > 0.0 && rds0.value() < 0.001 {
                rds0 = R::constant(0.0);
            }
            for overlap in [&mut cgso, &mut cgdo, &mut cgbo] {
                if overlap.value() < 0.0 {
                    *overlap = R::constant(0.0);
                }
            }
        }
        let overlaps = Overlaps {
            cgso: (cgso + binned.cf) * weff_cv,
            cgdo: (cgdo + binned.cf) * weff_cv,
            cgbo: cgbo * leff_cv,
            cgsl: binned.cgsl * weff_cv,
            cgdl: binned.cgdl * weff_cv,
            ckappa: binned.ckappa,
        };

        // Concentrations per cubic centimeter, as `b3temp.c` and `b3ld.c` read them.
        let mut npeak = binned.nch * 1e-6;
        if self.nch.is_none()
            && let Some(gamma1) = binned.gamma1
        {
            let product = gamma1 * cox;
            npeak = product * product * 3.021e22;
        }
        let phi = (npeak / ni).ln() * vtm0 * 2.0;
        check(phi.value() <= 0.0, || {
            format!(
                "the surface potential, 2 · Vtm · ln(nch / ni), must be positive, not {:?}",
                phi.value()
            )
        })?;
        let sqrt_phi = phi.sqrt();
        let xdep0 = (R::constant(2.0 * EPSSI) / (npeak * (Q * 1e6))).sqrt() * sqrt_phi;
        let litl = (binned.xj * tox * 3.0).sqrt();
        let vbi = (npeak * 1e20 / (ni * ni)).ln() * vtm0;
        let cdep0 = (npeak * (Q * EPSSI * 1e6) / 2.0 / phi).sqrt();
        let ldeb = (vtm0 * EPSSI / (npeak * (Q * 1e6))).sqrt() / 3.0;
        let acde = binned.acde * (npeak / 2e16).powf(-0.25);
        let (k1, k2) = self.body(&binned, (phi, sqrt_phi), (npeak, cox));
        let mut vbm = binned.vbm;
        if self.k1.is_none() && self.k2.is_none() && vbm.value() > 0.0 {
            vbm = -vbm;
        }
        let mut vbsc = if k2.value() < 0.0 {
            let t0 = k1 * 0.5 / k2;
            let vbsc = (phi - t0 * t0) * 0.9;
            if vbsc.value() > -3.0 {
                R::constant(-3.0)
            } else if vbsc.value() < -30.0 {
                R::constant(-30.0)
            } else {
                vbsc
            }
        } else {
            R::constant(-30.0)
        };
        if vbsc.value() > vbm.value() {
            vbsc = vbm;
        }
        let vfb = binned.vfb.unwrap_or_else(|| {
            binned
                .vth0
                .map_or(R::constant(-1.0), |vth0| vth0 * sign - phi - k1 * sqrt_phi)
        });
        let vth0 = binned
            .vth0
            .unwrap_or_else(|| (vfb + phi + k1 * sqrt_phi) * sign);
        let k1ox = k1 * tox / toxm;
        let k2ox = k2 * tox / toxm;
        let t1 = (tox * (EPSSI / EPSOX) * xdep0).sqrt();
        let falloff = |coefficient: R| {
            let t0 = (-(coefficient * leff * 0.5) / t1).exp();
            t0 + t0 * t0 * 2.0
        };
        let theta0vb0 = falloff(binned.dsub);
        let theta_rout = binned.pdiblc1 * falloff(binned.drout) + binned.pdiblc2;
        let tmp1 = vbi - phi;
        let tmp2 = factor1 * xdep0.sqrt();
        let short = |exponent: R| {
            let t1 = if exponent.value() > -EXP_THRESHOLD {
                exponent.exp()
            } else {
                R::constant(MIN_EXP)
            };
            t1 * (t1 * 2.0 + 1.0)
        };
        let narrow = binned.dvt0w * short(-(binned.dvt1w * weff * leff * 0.5) / tmp2) * tmp1;
        let channel = binned.dvt0 * short(-(binned.dvt1 * leff * 0.5) / tmp2) * tmp1;
        let t4 = tox * phi / (weff + binned.w0);
        let t5 = k1ox * ((binned.nlx / leff + 1.0).sqrt() - 1.0) * sqrt_phi
            + (binned.kt1 + binned.kt1l / leff) * rise;
        let vfbzb = vth0 * sign - narrow - channel + binned.k3 * t4 + t5 - phi - k1 * sqrt_phi;
        let delvto = self.threshold_adjustment;
        let core = Core {
            mobmod: self.mobility_model,
            capmod: self.capacitance_model,
            xpart: self.charge_partition,
            sign,
            tox,
            cox,
            factor1,
            vtm,
            temperature_rise: rise,
            leff,
            weff,
            leff_cv,
            weff_cv,
            vbsc,
            phi,
            sqrt_phi,
            phis3: sqrt_phi * phi,
            xdep0,
            vbi,
            cdep0,
            litl,
            ldeb,
            acde,
            k1,
            k1ox,
            k2ox,
            theta0vb0,
            theta_rout,
            abulk_cv_factor,
            u0temp: u0temp * self.mobility_multiplier,
            vsattemp,
            rds0,
            ua,
            ub,
            uc,
            vth0: vth0 + delvto,
            vfb: vfb + delvto * sign,
            vfbzb: vfbzb + delvto * sign,
            ngate: binned.ngate * 1e-6,
            binned,
        };
        let junctions = self.junctions((temperature, tnom), (vtm, vtm0), (eg0, weff))?;
        let lintnoi = self.lintnoi;
        check(lintnoi.value() > leff.value() / 2.0, || {
            format!(
                "lintnoi, {:?}, must be at most half the effective length, {:?}",
                lintnoi.value(),
                leff.value()
            )
        })?;
        let noise = Noise {
            model: self.noise_model,
            noia: self
                .noia
                .map_or(by_channel(self.channel, 1e20, 9.9e18), Real::value),
            noib: self
                .noib
                .map_or(by_channel(self.channel, 5e4, 2.4e3), Real::value),
            noic: self
                .noic
                .map_or(by_channel(self.channel, -1.4e-12, 1.4e-12), Real::value),
            em: self.em.value(),
            ef: self.ef.value(),
            af: self.af.value(),
            kf: self.kf.value(),
            length: (leff - lintnoi * 2.0).value(),
        };
        let resistances = self.resistances()?;
        Ok(PreparedBsim3 {
            core,
            junctions,
            overlaps,
            noise,
            series: resistances.map(series_value),
            multiplier,
            temperature: temperature.value(),
        })
    }

    /// The drain and source resistances, `rsh` times each diffusion's squares, of all the
    /// parallel copies together.
    pub(super) fn resistances(&self) -> Result<[R; 2], ParameterError> {
        let sheet = ParameterError::non_negative("rsh", self.rsh)?;
        let multiplier = ParameterError::positive("multiplier", self.multiplier)?;
        let drain = ParameterError::non_negative("drain_squares", self.drain_squares)?;
        let source = ParameterError::non_negative("source_squares", self.source_squares)?;
        Ok([drain, source].map(|squares| sheet * squares / multiplier))
    }

    /// The effective length and width of the drain current and of the charges, `Leff`,
    /// `Weff`, `LeffCV` and `WeffCV`.
    fn geometry(&self) -> Result<[R; 4], ParameterError> {
        let length = ParameterError::positive("length", self.length)?;
        let width = ParameterError::positive("width", self.width)?;
        let [by_length, by_width] = [length.pow(self.lln), width.pow(self.lwn)];
        let [of_length, of_width] = [length.pow(self.wln), width.pow(self.wwn)];
        let dl = lost(
            self.lint,
            [self.ll, self.lw, self.lwl],
            [by_length, by_width],
        );
        let dlc = lost(
            self.dlc.unwrap_or(self.lint),
            [
                self.llc.unwrap_or(self.ll),
                self.lwc.unwrap_or(self.lw),
                self.lwlc.unwrap_or(self.lwl),
            ],
            [by_length, by_width],
        );
        let dw = lost(
            self.wint,
            [self.wl, self.ww, self.wwl],
            [of_length, of_width],
        );
        let dwc = lost(
            self.dwc.unwrap_or(self.wint),
            [
                self.wlc.unwrap_or(self.wl),
                self.wwc.unwrap_or(self.ww),
                self.wwlc.unwrap_or(self.wwl),
            ],
            [of_length, of_width],
        );
        let effective = [
            ("effective length", length + self.xl - dl * 2.0),
            ("effective width", width + self.xw - dw * 2.0),
            (
                "effective length of the charges",
                length + self.xl - dlc * 2.0,
            ),
            (
                "effective width of the charges",
                width + self.xw - dwc * 2.0,
            ),
        ];
        let mut found = [R::constant(0.0); 4];
        for (value, (name, extent)) in found.iter_mut().zip(effective) {
            check(extent.value() <= 0.0, || {
                format!("the {name} must be positive, not {:?}", extent.value())
            })?;
            *value = extent;
        }
        Ok(found)
    }

    /// The fatal errors `b3check.c` stops on, of the `binned` parameters of a channel of
    /// effective length and width `leff` and `weff`, with the mobility and saturation
    /// velocity at the temperature.
    fn checked(
        &self,
        binned: &Binned<R>,
        [leff, weff]: [R; 2],
        [u0temp, vsattemp]: [R; 2],
    ) -> Result<(), ParameterError> {
        let value = |value: R| value.value();
        let (leff, weff) = (value(leff), value(weff));
        check(value(binned.nlx) < -leff, || {
            format!(
                "nlx, {:?}, must be at least the negative of the effective length, {leff:?}",
                value(binned.nlx)
            )
        })?;
        let positive = [
            ("nch", binned.nch),
            ("nsub", binned.nsub),
            ("xj", binned.xj),
            ("pclm", binned.pclm),
        ];
        for (name, number) in positive {
            check(value(number) <= 0.0, || {
                format!("{name} must be positive, not {:?}", value(number))
            })?;
        }
        let non_negative = [
            ("ngate", binned.ngate),
            ("dvt1", binned.dvt1),
            ("dvt1w", binned.dvt1w),
            ("dsub", binned.dsub),
            ("delta", binned.delta),
            ("drout", binned.drout),
            ("clc", binned.clc),
            ("ijth", self.ijth),
        ];
        for (name, number) in non_negative {
            check(value(number) < 0.0, || {
                format!("{name} must be at least 0, not {:?}", value(number))
            })?;
        }
        check(value(binned.ngate) > 1e31, || {
            format!(
                "ngate must be at most 1e31 per cubic meter, not {:?}",
                value(binned.ngate)
            )
        })?;
        for (name, offset) in [("w0", binned.w0), ("b1", binned.b1)] {
            #[expect(clippy::float_cmp, reason = "b3check.c's own test of a zero divisor")]
            let zero = value(offset) == -weff;
            check(zero, || {
                format!("{name} + Weff must not be zero, as {name} is {weff:?} below 0")
            })?;
        }
        check(value(u0temp) <= 0.0, || {
            format!(
                "u0 at the temperature must be positive, not {:?}",
                value(u0temp)
            )
        })?;
        check(value(vsattemp) <= 0.0, || {
            format!(
                "vsat at the temperature must be positive, not {:?}",
                value(vsattemp)
            )
        })
    }

    /// `cgso`, `cgdo` and `cgbo`, given or as `b3set.c` derives them from the oxide
    /// capacitance per area `cox`.
    #[expect(clippy::similar_names, reason = "the overlaps' names in BSIM3")]
    fn overlap_defaults(&self, cox: R) -> (R, R, R) {
        let derived = |lightly: R| match self.dlc {
            Some(dlc) if dlc.value() > 0.0 => dlc * cox - lightly,
            _ => self.xj * cox * 0.6,
        };
        let cgso = self.cgso.unwrap_or_else(|| derived(self.cgsl));
        let cgdo = self.cgdo.unwrap_or_else(|| derived(self.cgdl));
        let cgbo = self
            .cgbo
            .unwrap_or_else(|| self.dwc.unwrap_or(self.wint) * cox * 2.0);
        (cgso, cgdo, cgbo)
    }

    /// `k1` and `k2`: as given, the one missing at ngspice's default, or derived from the
    /// body effects at the surface and in the bulk, with `phi` and its root, the channel's
    /// doping `npeak` per cubic centimeter and the oxide capacitance `cox`.
    fn body(&self, binned: &Binned<R>, (phi, sqrt_phi): (R, R), (npeak, cox): (R, R)) -> (R, R) {
        if self.k1.is_some() || self.k2.is_some() {
            return (
                binned.k1.unwrap_or(R::constant(0.53)),
                binned.k2.unwrap_or(R::constant(-0.0186)),
            );
        }
        let mut vbx = binned
            .vbx
            .unwrap_or_else(|| phi - npeak * binned.xt * binned.xt * 7.7348e-4);
        if vbx.value() > 0.0 {
            vbx = -vbx;
        }
        let mut vbm = binned.vbm;
        if vbm.value() > 0.0 {
            vbm = -vbm;
        }
        let gamma1 = binned
            .gamma1
            .unwrap_or_else(|| npeak.sqrt() * 5.753e-12 / cox);
        let gamma2 = binned
            .gamma2
            .unwrap_or_else(|| (binned.nsub * 1e-6).sqrt() * 5.753e-12 / cox);
        let t0 = gamma1 - gamma2;
        let t1 = (phi - vbx).sqrt() - sqrt_phi;
        let t2 = (phi * (phi - vbm)).sqrt() - phi;
        let k2 = t0 * t1 / (t2 * 2.0 + vbm);
        let k1 = gamma2 - k2 * (phi - vbm).sqrt() * 2.0;
        (k1, k2)
    }

    /// The junctions at `temperature`, with the parameters' nominal temperature `tnom`,
    /// the thermal voltages at both, `vtm` and `vtm0`, silicon's band gap at the nominal
    /// temperature, `eg0`, and the effective width `weff`.
    fn junctions(
        &self,
        (temperature, tnom): (R, R),
        (vtm, vtm0): (R, R),
        (eg0, weff): (R, R),
    ) -> Result<Junctions<R>, ParameterError> {
        let nj = ParameterError::positive("nj", self.nj)?;
        let eg = -(temperature * temperature * 7.02e-4 / (temperature + 1108.0)) + 1.16;
        let t0 = eg0 / vtm0 - eg / vtm + self.xti * (temperature / tnom).ln();
        let scaling = (t0 / nj).exp();
        let floor = |value: R| value.max(R::constant(0.0));
        let js = floor(self.js * scaling);
        let jsw = floor(self.jsw * scaling);
        let difference = temperature - tnom;
        let scaled = |capacitance: R, coefficient: R| {
            let t0 = coefficient * difference;
            if t0.value() >= -1.0 {
                capacitance * (t0 + 1.0)
            } else {
                R::constant(0.0)
            }
        };
        let cj = scaled(self.cj, self.tcj);
        let cjsw = scaled(self.cjsw, self.tcjsw);
        let cjswg = scaled(self.cjswg.unwrap_or(self.cjsw), self.tcjswg);
        // `b3temp.c` raises a built-in potential below 0.1 V to it, then one that the
        // temperature brings below 0.01 V to that.
        let potential = |given: R, coefficient: R| {
            let given = given.max(R::constant(0.1));
            (given - coefficient * difference).max(R::constant(0.01))
        };
        let pbsw = self.pbsw;
        let potentials = [
            potential(self.pb, self.tpb),
            potential(pbsw, self.tpbsw),
            potential(self.pbswg.unwrap_or(pbsw), self.tpbswg),
        ];
        let mjsw = self.mjsw;
        let gradings = [self.mj, mjsw, self.mjswg.unwrap_or(mjsw)];
        for (name, grading) in ["mj", "mjsw", "mjswg"].into_iter().zip(gradings) {
            check(grading.value() >= 1.0 || grading.value() < 0.0, || {
                format!(
                    "{name} must be at least 0 and below 1, not {:?}",
                    grading.value()
                )
            })?;
        }
        let slope = vtm * nj;
        let ijth = self.ijth;
        let junction = |area: R, perimeter: R| {
            let saturation_current = if area.value() <= 0.0 && perimeter.value() <= 0.0 {
                R::constant(1e-14)
            } else {
                area * js + perimeter * jsw
            };
            let linear = (saturation_current.value() > 0.0 && ijth.value() > 0.0).then(|| {
                let vjm = slope * (ijth / saturation_current + 1.0).ln();
                (vjm, saturation_current * (vjm / slope).exp())
            });
            let (sidewall, gate) = if perimeter.value() < weff.value() {
                (R::constant(0.0), cjswg * perimeter)
            } else {
                (cjsw * (perimeter - weff), cjswg * weff)
            };
            Junction {
                saturation_current,
                linear,
                capacitances: [cj * area, sidewall, gate],
            }
        };
        Ok(Junctions {
            slope,
            drain: junction(self.drain_area, self.drain_perimeter),
            source: junction(self.source_area, self.source_perimeter),
            potentials,
            gradings,
        })
    }
}
