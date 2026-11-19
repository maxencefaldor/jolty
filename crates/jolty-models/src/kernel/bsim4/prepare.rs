//! A BSIM4 transistor's numbers at its temperature and size, as ngspice 47's
//! `b4v5temp.c` prepares them, with `b4v5geo.c`'s diffusions, checked as its
//! `b4v5check.c` checks them.
//!
//! Part of jolty's BSIM4, derived from BSIM4.5.0 of the UC Berkeley BSIM Research Group:
//! Copyright 2005 Regents of the University of California. All rights reserved. Used
//! under the BSIM4 terms of use, which the repository's `NOTICE` reproduces.

use super::channel::theta;
use super::charges::Depletion;
use super::dexp;
use super::parameters::{Binned, Bsim4Parameters};
use super::{
    BodyResistance, DrainSourceResistance, EPS0, FlickerNoise, GateResistance, PI, Perimeter,
    ThermalNoise,
};
use crate::kernel::bsim::{
    D, EPSSI, KBOQ, Q, band_gap, by_channel, check, default_overlaps, derived_body_factors,
    heated_capacitance, heated_potential, intrinsic_carrier, lost, vbsc_clamp,
};
use crate::kernel::parts::series_value;
use crate::{ParameterError, Real};

/// The smoothing `b4v5temp.c` adds to the stress effect's factors, its `DELTA`.
const STRESS_DELTA: f64 = 1.0e-9;
/// The oxide thickness `b4v5set.c` defaults `toxe` to, in meters.
const TOXE: f64 = 30.0e-10;

/// The numbers the drain current reads, its channel's size and temperature folded in.
/// Concentrations are per cubic centimeter, as the equations of `b4v5ld.c` take them.
#[derive(Debug, Clone, Copy)]
pub(super) struct Core<R: Real> {
    /// The channel's sign, 1 for an n-channel and -1 for a p-channel.
    pub(super) sign: f64,
    pub(super) toxe: R,
    pub(super) toxp: R,
    pub(super) coxe: R,
    pub(super) coxp: R,
    /// Oxide-to-silicon depletion factor `√(εsi / (epsrox · ε0) · toxe)`.
    pub(super) factor1: R,
    /// Thermal voltage `k·T / q` at device temperature using BSIM4's `k / q`.
    pub(super) vtm: R,
    /// Relative temperature rise `T / Tnom - 1`.
    pub(super) temperature_rise: R,
    pub(super) leff: R,
    pub(super) weff: R,
    pub(super) weff_cj: R,
    pub(super) leff_cv: R,
    pub(super) weff_cv: R,
    pub(super) fingers: R,
    /// The drain's share of the channel's charge in saturation, `xpart`.
    pub(super) xpart: f64,
    pub(super) abulk_cv_factor: R,
    /// The flat-band voltage at zero bias of the charges.
    pub(super) vfbzb: R,
    /// The Debye length of the charge-thickness model, and its exponential coefficient
    /// `acde` scaled by the doping.
    pub(super) ldeb: R,
    pub(super) acde: R,
    pub(super) phi: R,
    pub(super) sqrt_phi: R,
    pub(super) xdep0: R,
    pub(super) vbi: R,
    pub(super) cdep0: R,
    pub(super) litl: R,
    pub(super) k1: R,
    pub(super) k1ox: R,
    pub(super) k2ox: R,
    pub(super) mstar: R,
    pub(super) voffcbn: R,
    pub(super) theta0vb0: R,
    pub(super) theta_rout: R,
    /// Internal channel resistance from `rdsw` (zero when `rdsmod` is 1).
    pub(super) rds0: R,
    pub(super) rdswmin: R,
    pub(super) ua: R,
    pub(super) ub: R,
    pub(super) uc: R,
    pub(super) ud: R,
    pub(super) u0temp: R,
    pub(super) vsattemp: R,
    /// Source-end velocity-limit factor `(1 - T0) / (1 + T0)` when `vtl > 0`.
    pub(super) tfactor: Option<R>,
    /// The threshold at zero bias of a long, wide channel, negative for a p-channel as the
    /// card's `vth0` is, the stress effect and `delvto` added.
    pub(super) vth0: R,
    pub(super) vfb: R,
    pub(super) eta0: R,
    pub(super) vbsc: R,
    pub(super) vtfbphi2: R,
    /// The poly gate's doping, per cubic centimeter.
    pub(super) ngate: R,
    pub(super) pditsl: R,
    /// The velocity overshoot's binned `lambda`, where a positive one is given.
    pub(super) lambda: Option<R>,
    pub(super) binned: Binned<R>,
}

/// One junction between a body node and the drain or the source.
#[derive(Debug, Clone, Copy)]
pub(super) struct Junction<R: Real> {
    pub(super) saturation_current: R,
    /// Diode thermal slope `Nvtm`, the emission coefficient times thermal voltage.
    pub(super) slope: R,
    /// The voltage above which the current continues linearly, and the current
    /// `Is · e^(vjm / Nvtm)` there.
    pub(super) vjm: R,
    pub(super) current_at_vjm: R,
    /// The trap-assisted tunneling of the bottom, the isolation sidewall and the gate
    /// sidewall: each one's saturation current, its `vts` and its slope `Nvtm0 · njts`.
    pub(super) tunneling: [(R, R, R); 3],
}

/// The overlaps between the gate and the source, the drain and the body.
#[derive(Debug, Clone, Copy)]
pub(super) struct Overlaps<R: Real> {
    /// Total gate-source and gate-drain overlap capacitances `(cgso + cf) · WeffCV` and
    /// `(cgdo + cf) · WeffCV`.
    pub(super) cgso: R,
    pub(super) cgdo: R,
    /// Lightly-doped gate-source and gate-drain overlap capacitances `cgsl · WeffCV` and
    /// `cgdl · WeffCV`.
    pub(super) cgsl: R,
    pub(super) cgdl: R,
    pub(super) ckappas: R,
    pub(super) ckappad: R,
    /// Total gate-body overlap capacitance `cgbo · LeffCV · nf`.
    pub(super) cgbo: R,
    pub(super) fingers: R,
}

/// Bias-dependent external drain and source resistances for `rdsmod` 1. Stores zero- and
/// high-gate-bias terms (`rd0`, `rdwmin`, `rs0`, `rswmin`), diffusion flat-band voltage
/// `vfbsd`, bias coefficients `prwg` and `prwb`, and sheet conductances (`1 kS` when `nrd`
/// or `nrs` is omitted per `b4v5temp.c`).
#[derive(Debug, Clone, Copy)]
pub(super) struct External<R: Real> {
    pub(super) minimums: [R; 2],
    pub(super) resistances: [R; 2],
    pub(super) vfbsd: R,
    pub(super) prwg: R,
    pub(super) prwb: R,
    pub(super) sheet: [R; 2],
}

/// Gate electrode and intrinsic-input resistance parameters for `rgatemod` 1 to 3.
/// Intrinsic-input conductance follows
/// `xrcrg1 · (xrcrg2 · vtm · β + Ids / Vdseff) · nf` from single-finger `β` and
/// `Ids / Vdseff` (`b4v5ld.c`, lines 1999 to 2019).
#[derive(Debug, Clone, Copy)]
pub(super) struct Gate<R: Real> {
    pub(super) model: GateResistance,
    /// Single-copy gate electrode conductance `grgeltd`.
    pub(super) electrode: R,
    /// Finger-scaled intrinsic-input factor `xrcrg1 · nf`.
    pub(super) xrcrg1_nf: R,
    /// Temperature-scaled intrinsic-input factor `xrcrg2 · vtm`.
    pub(super) xrcrg2_vtm: R,
}

/// The numbers of the noise models.
#[derive(Debug, Clone, Copy)]
pub(super) struct Noise {
    pub(super) flicker_model: FlickerNoise,
    pub(super) thermal_model: ThermalNoise,
    pub(super) noia: f64,
    pub(super) noib: f64,
    pub(super) noic: f64,
    pub(super) em: f64,
    pub(super) ef: f64,
    pub(super) af: f64,
    pub(super) kf: f64,
    pub(super) ntnoi: f64,
    pub(super) tnoia: f64,
    pub(super) tnoib: f64,
    pub(super) rnoia: f64,
    pub(super) rnoib: f64,
    /// The effective length less twice `lintnoi`.
    pub(super) length: f64,
}

/// A transistor's numbers at its temperature.
#[derive(Debug, Clone, Copy)]
pub struct PreparedBsim4<R: Real> {
    pub(super) core: Core<R>,
    pub(super) overlaps: Overlaps<R>,
    pub(super) drain_depletion: Depletion<R>,
    pub(super) source_depletion: Depletion<R>,
    pub(super) drain_junction: Junction<R>,
    pub(super) source_junction: Junction<R>,
    /// The drain and source resistances' stamped values: each conductance, or each
    /// resistance where its current is an unknown, of the parallel copies together.
    pub(super) series: [R; 2],
    pub(super) external: Option<External<R>>,
    /// The drain and source resistances, of the parallel copies together, none where
    /// there is no node behind them.
    pub(super) resistances: [R; 2],
    pub(super) noise: Noise,
    pub(super) temperature: f64,
    /// The conductances of the five body resistances of `rbodymod` 1, in the order of
    /// `BODY_RESISTANCES`, of the parallel copies together.
    pub(super) body_network: Option<[R; 5]>,
    /// The gate's resistances, none at `rgatemod` 0.
    pub(super) gate: Option<Gate<R>>,
    pub(super) multiplier: R,
}

/// Temperature-derived quantities shared across `b4v5temp.c`'s preparation stages.
#[derive(Debug, Clone, Copy)]
struct Thermal<R: Real> {
    ratio: R,
    rise: R,
    delta: R,
    vtm0: R,
    vtm: R,
    eg0: R,
    eg: R,
    ni: R,
}

/// Drawn and effective channel dimensions of one finger.
#[derive(Debug, Clone, Copy)]
struct Geometry<R: Real> {
    length: R,
    l_new: R,
    w_new: R,
    dwc_model: R,
    leff: R,
    weff: R,
    leff_cv: R,
    weff_cv: R,
    weff_cj: R,
}

/// Source and drain saturation densities, emission coefficients and forward thresholds.
type JunctionScalings<R> = ([[R; 3]; 2], [R; 2], [R; 2]);

/// Computes `b4v5temp.c`'s thermal ratios, band gaps and intrinsic carrier concentration.
fn thermal_state<R: Real>(temperature: R, nominal: R) -> Thermal<R> {
    let ratio = temperature / nominal;
    let rise = ratio - 1.0;
    let delta = temperature - nominal;
    let nominal_thermal_voltage = nominal * KBOQ;
    let nominal_band_gap = band_gap(nominal);
    let intrinsic = intrinsic_carrier(nominal, nominal_band_gap, nominal_thermal_voltage);
    Thermal {
        ratio,
        rise,
        delta,
        vtm0: nominal_thermal_voltage,
        vtm: temperature * KBOQ,
        eg0: nominal_band_gap,
        eg: band_gap(temperature),
        ni: intrinsic,
    }
}

/// Builds one `diomod` 1 [`Junction`] from its saturation current, emission coefficient,
/// forward threshold and trap-assisted tunneling terms.
fn build_junction<R: Real>(
    vtm: R,
    saturation_current: R,
    emission: R,
    ijth: R,
    tunneling: [(R, R, R); 3],
) -> Junction<R> {
    let slope = vtm * emission;
    let vjm = slope * (ijth / saturation_current + 1.0).ln();
    Junction {
        saturation_current,
        slope,
        vjm,
        current_at_vjm: saturation_current * (vjm / slope).exp(),
        tunneling,
    }
}

/// Computes source and drain diffusion perimeters and areas for `fingers` fingers from
/// `weff_cj` and spacings `[dmcg, dmci]`, matching `BSIM4v5PAeffGeo` (`geomod = 0`, `min = 0`).
#[expect(clippy::similar_names, reason = "b4v5geo.c's names of the diffusions")]
fn diffusions<R: Real>(fingers: f64, weff_cj: R, [dmcg, dmci]: [R; 2]) -> [R; 4] {
    // `BSIM4v5NumFingerDiff`: an odd number of fingers ends in a source and a drain, an
    // even one in two drains with `min` 0.
    let whole = fingers.trunc();
    let (end_drain, inner_drain, end_source, inner_source) = if whole % 2.0 == 0.0 {
        (0.0, fingers, 2.0, 2.0 * (fingers / 2.0 - 1.0).max(0.0))
    } else {
        let inner = 2.0 * ((fingers - 1.0) / 2.0).max(0.0);
        (1.0, inner, 1.0, inner)
    };
    let t0 = dmcg + dmci;
    let isolated_perimeter = t0 + t0 + weff_cj;
    let shared_perimeter = dmcg + dmcg;
    let isolated_area = t0 * weff_cj;
    let shared_area = dmcg * weff_cj;
    [
        isolated_perimeter * end_source + shared_perimeter * inner_source,
        isolated_perimeter * end_drain + shared_perimeter * inner_drain,
        isolated_area * end_source + shared_area * inner_source,
        isolated_area * end_drain + shared_area * inner_drain,
    ]
}

impl<R: Real> Bsim4Parameters<R> {
    /// Resolves electrical and physical oxide thicknesses `[toxe, toxp]` from each other
    /// and `dtox`, matching `b4v5temp.c`.
    #[expect(
        clippy::similar_names,
        reason = "the oxide thicknesses' names in BSIM4"
    )]
    fn thicknesses(&self) -> [R; 2] {
        match (self.toxe, self.toxp) {
            (Some(toxe), None) => [toxe, toxe - self.dtox],
            (None, Some(toxp)) => [toxp + self.dtox, toxp],
            (Some(toxe), Some(toxp)) => [toxe, toxp],
            (None, None) => [R::constant(TOXE); 2],
        }
    }

    /// Returns the card `toxe` or the 3 nm `b4v5set.c` default before `b4v5temp.c`
    /// resolves it against `toxp`.
    pub(super) fn set_toxe(&self) -> R {
        self.toxe.unwrap_or(R::constant(TOXE))
    }

    /// Computes combined parallel drain and source sheet resistances from `rsh`, `nrd`,
    /// and `nrs` for `rdsmod = 0` and `rgeomod = 0` (`b4v5set.c`).
    pub(super) fn resistances(&self) -> Result<[R; 2], ParameterError> {
        let sheet = self.rsh;
        let multiplier = ParameterError::positive("multiplier", self.multiplier)?;
        Ok(
            [self.drain_squares, self.source_squares].map(|squares| match squares {
                Some(squares) if sheet.value() > 0.0 && squares.value() > 0.0 => {
                    sheet * squares / multiplier
                }
                _ => R::constant(0.0),
            }),
        )
    }

    /// Prepares temperature- and geometry-dependent BSIM4 quantities at `temperature` in
    /// kelvins.
    pub(super) fn prepared(&self, temperature: R) -> Result<PreparedBsim4<R>, ParameterError> {
        if temperature.value() <= 0.0 {
            return Err(ParameterError(format!(
                "a BSIM4 transistor's temperature lies above 0 K, not at {:?} K",
                temperature.value()
            )));
        }
        let tnom = ParameterError::positive("nominal_temperature", self.nominal_temperature)?;
        let multiplier = ParameterError::positive("multiplier", self.multiplier)?;
        let fingers = self.fingers;
        check(fingers.value() < 1.0, || {
            format!("nf is at least 1, not {:?}", fingers.value())
        })?;
        let oxide = self.oxide()?;
        let thermal = thermal_state(temperature, tnom);
        let scalings = self.saturation_densities(&thermal)?;
        let geometry = self.geometry(fingers)?;
        let (mut core, [ndep, nsub, nsd, pow_weff_wr]) =
            self.prepare_core(&thermal, &geometry, oxide, fingers)?;
        let body_network = self.body_network(multiplier);
        let gate = self.gate_network(&core.binned, &thermal, &geometry, fingers);
        let areas = self.diffusion_areas(&geometry, fingers);
        let resistances = self.resistances()?;
        let series = resistances.map(series_value);
        let [drain_junction, source_junction] = self.junctions(&thermal, scalings, areas)?;

        // `b4v5check.c` stops on these whatever the gate's model.
        check(geometry.l_new.value() <= self.xgl.value(), || {
            format!(
                "xgl, {:?}, is less than the drawn length plus xl, {:?}",
                self.xgl.value(),
                geometry.l_new.value()
            )
        })?;
        check(self.ngcon.value() < 1.0, || {
            format!("ngcon is at least 1, not {:?}", self.ngcon.value())
        })?;
        self.checked(
            &core.binned,
            [geometry.leff, geometry.weff],
            [ndep, nsub, core.phi, core.ngate],
            [core.u0temp, core.vsattemp],
        )?;
        let external = self.clamp_and_external(&mut core, &thermal, pow_weff_wr, nsd);
        let overlaps = self.overlaps(&core.binned, core.coxe, &geometry, fingers);
        let [drain_depletion, source_depletion] = self.depletions(&thermal, areas);
        let noise = self.noise_parameters(&geometry);
        Ok(PreparedBsim4 {
            external,
            overlaps,
            resistances,
            noise,
            temperature: temperature.value(),
            drain_depletion,
            source_depletion,
            core,
            drain_junction,
            source_junction,
            series,
            body_network,
            gate,
            multiplier,
        })
    }

    /// Validates oxide thicknesses and returns `[[toxe, toxp, toxm], [coxe, coxp, factor1]]`.
    fn oxide(&self) -> Result<[[R; 3]; 2], ParameterError> {
        let [electrical, physical] = self.thicknesses();
        let measured = self.toxm.unwrap_or(self.set_toxe());
        for (name, value) in [
            ("toxe", electrical),
            ("toxp", physical),
            ("toxm", measured),
            ("toxref", self.toxref),
        ] {
            ParameterError::positive(name, value)?;
        }
        Ok([
            [electrical, physical, measured],
            [
                self.epsrox * EPS0 / electrical,
                self.epsrox * EPS0 / physical,
                (electrical * EPSSI / (self.epsrox * EPS0)).sqrt(),
            ],
        ])
    }

    /// Scales source and drain saturation current densities, emission coefficients and
    /// forward thresholds to `thermal`'s temperature, the drain's defaulting to the
    /// source's.
    fn saturation_densities(
        &self,
        thermal: &Thermal<R>,
    ) -> Result<JunctionScalings<R>, ParameterError> {
        let source_raw = [self.jss, self.jsws, self.jswgs];
        let drain_raw = [
            self.jsd.unwrap_or(self.jss),
            self.jswd.unwrap_or(self.jsws),
            self.jswgd.unwrap_or(self.jswgs),
        ];
        let source_emission = ParameterError::positive("njs", self.njs)?;
        let drain_emission = match self.njd {
            Some(njd) => ParameterError::positive("njd", njd)?,
            None => source_emission,
        };
        let heated = |densities: [R; 3], exponent: R, emission: R| {
            let t0 = thermal.eg0 / thermal.vtm0 - thermal.eg / thermal.vtm;
            let t3 = ((t0 + exponent * thermal.ratio.ln()) / emission).exp();
            densities.map(|density| {
                let scaled = density * t3;
                if scaled.value() < 0.0 {
                    R::constant(0.0)
                } else {
                    scaled
                }
            })
        };
        let at_least = |value: R| {
            if value.value() <= 0.0 {
                R::constant(0.1)
            } else {
                value
            }
        };
        Ok((
            [
                heated(source_raw, self.xtis, source_emission),
                heated(drain_raw, self.xtid.unwrap_or(self.xtis), drain_emission),
            ],
            [source_emission, drain_emission],
            [
                at_least(self.ijthsfwd),
                at_least(self.ijthdfwd.unwrap_or(self.ijthsfwd)),
            ],
        ))
    }

    /// Computes effective channel, charge and junction dimensions of one finger.
    fn geometry(&self, fingers: R) -> Result<Geometry<R>, ParameterError> {
        let length = ParameterError::positive("length", self.length)?;
        let width = ParameterError::positive("width", self.width)?;
        let drawn_length = length + self.xl;
        let drawn_width = width / fingers + self.xw;
        let [by_length, by_width] = [drawn_length.pow(self.lln), drawn_width.pow(self.lwn)];
        let [of_length, of_width] = [drawn_length.pow(self.wln), drawn_width.pow(self.wwn)];
        let delta_length = lost(
            self.lint,
            [self.ll, self.lw, self.lwl],
            [by_length, by_width],
        );
        let delta_width = lost(
            self.wint,
            [self.wl, self.ww, self.wwl],
            [of_length, of_width],
        );
        let dwc_model = self.dwc.unwrap_or(self.wint);
        let width_lost_cv = lost(
            R::constant(0.0),
            [
                self.wlc.unwrap_or(self.wl),
                self.wwc.unwrap_or(self.ww),
                self.wwlc.unwrap_or(self.wwl),
            ],
            [of_length, of_width],
        );
        let junction_width_lost = self.dwj.unwrap_or(dwc_model) + width_lost_cv;
        let length_lost_cv = lost(
            self.dlc.unwrap_or(self.lint),
            [
                self.llc.unwrap_or(self.ll),
                self.lwc.unwrap_or(self.lw),
                self.lwlc.unwrap_or(self.lwl),
            ],
            [by_length, by_width],
        );
        let charge_length = drawn_length - length_lost_cv * 2.0;
        let charge_width = drawn_width - (dwc_model + width_lost_cv) * 2.0;
        let effective_length = drawn_length - delta_length * 2.0;
        let effective_width = drawn_width - delta_width * 2.0;
        let junction_width = drawn_width - junction_width_lost * 2.0;
        for (name, value) in [
            ("length", effective_length),
            ("width", effective_width),
            ("length of the charges", charge_length),
            ("width of the charges", charge_width),
            ("width of the junctions", junction_width),
        ] {
            check(value.value() <= 0.0, || {
                format!("the effective {name} is positive, not {:?}", value.value())
            })?;
        }
        Ok(Geometry {
            length,
            l_new: drawn_length,
            w_new: drawn_width,
            dwc_model,
            leff: effective_length,
            weff: effective_width,
            leff_cv: charge_length,
            weff_cv: charge_width,
            weff_cj: junction_width,
        })
    }

    /// Bins parameters, applies `tempmod` 0 scaling, doping potentials, short-channel
    /// roll-offs and LOD stress, and returns `(core, [ndep, nsub, nsd, pow_weff_wr])`.
    fn prepare_core(
        &self,
        thermal: &Thermal<R>,
        geometry: &Geometry<R>,
        oxide: [[R; 3]; 2],
        fingers: R,
    ) -> Result<(Core<R>, [R; 4]), ParameterError> {
        let [thicknesses, capacitances] = oxide;
        let [electrical_oxide, physical_oxide, measured_oxide] = thicknesses;
        let [electrical_cap, physical_cap, factor1] = capacitances;
        let sign = self.channel.sign();
        let rise = thermal.rise;
        let (mut binned, [pow_weff_wr, u0temp], tfactor) =
            self.tempmod_zero(thermal, geometry, fingers);
        let t10 = binned.prt * rise;
        let mut vsattemp = binned.vsat - binned.at * rise;

        let (
            [channel, substrate, diffusion, gate],
            [phi, sqrt_phi, depletion_depth, built_in_potential],
            (k1, mut k2, vfb, mut vth0),
        ) = self.doping_potentials(&mut binned, thermal, electrical_cap);
        let k1ox = k1 * electrical_oxide / measured_oxide;
        let [theta0vb0, theta_rout, vfbzb_factor] = self.short_channel_factors(
            &binned,
            (thermal, geometry),
            (electrical_oxide, factor1, depletion_depth),
            (phi, sqrt_phi, built_in_potential, k1, k1ox),
        );
        let mut eta0 = binned.eta0;
        self.apply_stress(
            thermal,
            geometry,
            fingers,
            (&mut vth0, &mut k2, &mut eta0, &mut vsattemp),
        )?;
        vth0 += self.threshold_adjustment;
        let vfb = vfb + self.threshold_adjustment * sign;
        let vtfbphi2 = (vth0 * sign - vfb - phi).max(R::constant(0.0)) * 4.0;
        let core = Core {
            sign,
            toxe: electrical_oxide,
            toxp: physical_oxide,
            coxe: electrical_cap,
            coxp: physical_cap,
            factor1,
            vtm: thermal.vtm,
            temperature_rise: rise,
            leff: geometry.leff,
            weff: geometry.weff,
            weff_cj: geometry.weff_cj,
            leff_cv: geometry.leff_cv,
            weff_cv: geometry.weff_cv,
            fingers,
            xpart: self.charge_partition,
            abulk_cv_factor: (binned.clc / geometry.leff_cv).pow(binned.cle) + 1.0,
            vfbzb: vfbzb_factor + vth0 * sign,
            ldeb: (thermal.vtm0 * EPSSI / (channel * (Q * 1.0e6))).sqrt() / 3.0,
            acde: binned.acde * (channel / 2.0e16).powf(-0.25),
            phi,
            sqrt_phi,
            xdep0: depletion_depth,
            vbi: built_in_potential,
            cdep0: (channel * (Q * EPSSI * 1e6 / 2.0) / phi).sqrt(),
            litl: (binned.xj * electrical_oxide * 3.0).sqrt(),
            k1,
            k1ox,
            k2ox: k2 * electrical_oxide / measured_oxide,
            mstar: binned.minv.atan() / PI + 0.5,
            voffcbn: binned.voff + self.voffl / geometry.leff,
            theta0vb0,
            theta_rout,
            rds0: (binned.rdsw + t10) * fingers / pow_weff_wr,
            rdswmin: (self.rdswmin + t10) * fingers / pow_weff_wr,
            ua: binned.ua + binned.ua1 * rise,
            ub: binned.ub + binned.ub1 * rise,
            uc: binned.uc + binned.uc1 * rise,
            ud: binned.ud + binned.ud1 * rise,
            u0temp: u0temp * self.mobility_multiplier,
            vsattemp,
            tfactor,
            vth0,
            vfb,
            eta0,
            vbsc: vbsc_clamp(phi, k1, k2, binned.vbm),
            vtfbphi2,
            ngate: gate,
            pditsl: self.pditsl,
            lambda: self
                .lambda
                .is_some_and(|lambda| lambda.value() > 0.0)
                .then_some(binned.lambda),
            binned,
        };
        Ok((core, [channel, substrate, diffusion, pow_weff_wr]))
    }

    /// Bins parameters, applies `tempmod` 0 scaling to `voff`, and returns
    /// `(binned, [pow_weff_wr, u0temp], tfactor)`.
    fn tempmod_zero(
        &self,
        thermal: &Thermal<R>,
        geometry: &Geometry<R>,
        fingers: R,
    ) -> (Binned<R>, [R; 2], Option<R>) {
        let inverse = [
            geometry.leff.recip(),
            geometry.weff.recip(),
            (geometry.leff * geometry.weff).recip(),
        ];
        let mut binned = self.binned(inverse);
        let pow_weff_wr = (geometry.weff_cj * 1e6).pow(binned.wr) * fingers;
        let length_mobility = -(binned.up * (-(geometry.leff / binned.lp)).exp()) + 1.0;
        let u0temp = binned.u0 * length_mobility * thermal.ratio.pow(binned.ute);
        binned.voff = binned.voff * (binned.tvoff * thermal.delta + 1.0);
        let tfactor = self.vtl.is_some_and(|vtl| vtl.value() > 0.0).then(|| {
            let lc = if self.lc.value() < 0.0 {
                R::constant(0.0)
            } else {
                self.lc
            };
            let t0 = geometry.leff / (binned.xn * geometry.leff + lc);
            (-t0 + 1.0) / (t0 + 1.0)
        });
        (binned, [pow_weff_wr, u0temp], tfactor)
    }

    /// Computes doping concentrations (in cm⁻³), potentials, body factors and zero-bias
    /// threshold, normalizing positive `binned.vbm` when `k1` and `k2` are derived.
    fn doping_potentials(
        &self,
        binned: &mut Binned<R>,
        thermal: &Thermal<R>,
        electrical_cap: R,
    ) -> ([R; 4], [R; 4], (R, R, R, R)) {
        let sign = self.channel.sign();
        // Concentrations per cubic centimeter, as `b4v5temp.c` and `b4v5ld.c` read them.
        let mut channel = binned.ndep * 1e-6;
        if self.ndep.is_none()
            && let Some(gamma1) = binned.gamma1
        {
            let product = gamma1 * electrical_cap;
            channel = product * product * 3.012_48e22;
        }
        let substrate = binned.nsub * 1e-6;
        let diffusion = binned.nsd * 1e-6;
        let gate = binned.ngate * 1e-6;
        let phi = thermal.vtm0 * (channel / thermal.ni).ln() + binned.phin + 0.4;
        let sqrt_phi = phi.sqrt();
        let depletion_depth = (R::constant(2.0 * EPSSI / (Q * 1e6)) / channel).sqrt() * sqrt_phi;
        let built_in_potential =
            thermal.vtm0 * (diffusion * channel / (thermal.ni * thermal.ni)).ln();
        let (k1, k2) = if self.k1.is_some() || self.k2.is_some() {
            (
                binned.k1.unwrap_or(R::constant(0.53)),
                binned.k2.unwrap_or(R::constant(-0.0186)),
            )
        } else {
            let (k1, k2, vbm) = derived_body_factors(
                (phi, sqrt_phi),
                (channel, substrate, electrical_cap),
                (binned.vbx, binned.vbm, binned.xt),
                (binned.gamma1, binned.gamma2),
            );
            binned.vbm = vbm;
            (k1, k2)
        };
        let vfb = binned.vfb.unwrap_or_else(|| match binned.vth0 {
            Some(vth0) => vth0 * sign - phi - k1 * sqrt_phi,
            _ => R::constant(-1.0),
        });
        let vth0 = binned
            .vth0
            .unwrap_or_else(|| (vfb + phi + k1 * sqrt_phi) * sign);
        (
            [channel, substrate, diffusion, gate],
            [phi, sqrt_phi, depletion_depth, built_in_potential],
            (k1, k2, vfb, vth0),
        )
    }

    /// Computes `[theta0vb0, theta_rout, vfbzb_factor]` from short-channel roll-off terms.
    fn short_channel_factors(
        &self,
        binned: &Binned<R>,
        (thermal, geometry): (&Thermal<R>, &Geometry<R>),
        (electrical_oxide, factor1, depletion_depth): (R, R, R),
        (phi, sqrt_phi, built_in_potential, k1, k1ox): (R, R, R, R, R),
    ) -> [R; 3] {
        let characteristic_length =
            (electrical_oxide * depletion_depth * EPSSI / (self.epsrox * EPS0)).sqrt();
        let theta0vb0 = theta(D::constant(
            binned.dsub * geometry.leff / characteristic_length,
        ))
        .value;
        let theta_rout = binned.pdiblc1
            * theta(D::constant(
                binned.drout * geometry.leff / characteristic_length,
            ))
            .value
            + binned.pdiblc2;
        let short_channel_scale = factor1 * depletion_depth.sqrt();
        let v0 = built_in_potential - phi;
        let t8 = binned.dvt0w
            * theta(D::constant(
                binned.dvt1w * geometry.weff * geometry.leff / short_channel_scale,
            ))
            .value
            * v0;
        let t9 = binned.dvt0
            * theta(D::constant(
                binned.dvt1 * geometry.leff / short_channel_scale,
            ))
            .value
            * v0;
        let t4 = electrical_oxide * phi / (geometry.weff + binned.w0);
        let t0 = (binned.lpe0 / geometry.leff + 1.0).sqrt();
        let t3 = (binned.kt1 + binned.kt1l / geometry.leff) * thermal.rise;
        let t5 = k1ox * (t0 - 1.0) * sqrt_phi + t3;
        [
            theta0vb0,
            theta_rout,
            -t8 - t9 + binned.k3 * t4 + t5 - phi - k1 * sqrt_phi,
        ]
    }

    /// Applies length-of-diffusion stress shifts to `threshold`, `k2`, `eta0` and
    /// `vsattemp`.
    fn apply_stress(
        &self,
        thermal: &Thermal<R>,
        geometry: &Geometry<R>,
        fingers: R,
        (threshold, k2, eta0, vsattemp): (&mut R, &mut R, &mut R, &mut R),
    ) -> Result<(), ParameterError> {
        let width_offset = if self.wlod.value() < 0.0 {
            R::constant(0.0)
        } else {
            self.wlod
        };
        let stressed_width = geometry.w_new + width_offset;
        let ku0 = lost(
            R::constant(1.0),
            [self.lku0, self.wku0, self.pku0],
            [
                geometry.l_new.pow(self.llodku0),
                stressed_width.pow(self.wlodku0),
            ],
        );
        let kvth0 = lost(
            R::constant(1.0),
            [self.lkvth0, self.wkvth0, self.pkvth0],
            [
                geometry.l_new.pow(self.llodvth),
                stressed_width.pow(self.wlodvth),
            ],
        );
        let kvth0 = (kvth0 * kvth0 + STRESS_DELTA).sqrt();
        let ku0temp = ku0 * (self.tku0 * thermal.rise + 1.0) + STRESS_DELTA;
        let half = geometry.length * 0.5;
        let reference_od = (self.saref + half).recip() + (self.sbref + half).recip();
        let rho_ref = self.ku0 / ku0temp * reference_od;
        let (first_edge, second_edge) = (self.diffusion_edge_a, self.diffusion_edge_b);
        let spacing = self.finger_spacing.unwrap_or(self.dmcg * 2.0);
        #[expect(
            clippy::float_cmp,
            reason = "b4v5temp.c stresses a single finger, `nf` exactly 1, as it is"
        )]
        let single = fingers.value() == 1.0;
        let stressed = first_edge.value() > 0.0
            && second_edge.value() > 0.0
            && (single || (fingers.value() > 1.0 && spacing.value() > 0.0));
        if !stressed {
            return Ok(());
        }
        for (name, value) in [("saref", self.saref), ("sbref", self.sbref)] {
            ParameterError::positive(name, value)?;
        }
        let kvsat = self.kvsat.max(R::constant(-1.0)).min(R::constant(1.0));
        let mut inv_first = R::constant(0.0);
        let mut inv_second = R::constant(0.0);
        let mut index = 0.0;
        while index < fingers.value() {
            let step = (spacing + geometry.length) * index;
            inv_first += (fingers * (first_edge + half + step)).recip();
            inv_second += (fingers * (second_edge + half + step)).recip();
            index += 1.0;
        }
        let effective_od = inv_first + inv_second;
        let rho = self.ku0 / ku0temp * effective_od;
        // `b4v5temp.c` scales the mobility here, then takes the unstressed one with
        // `mulu0` below: the stress leaves the mobility as it was.
        *vsattemp = *vsattemp * (kvsat * rho + 1.0) / (kvsat * rho_ref + 1.0);
        let offset = effective_od - reference_od;
        *threshold += self.kvth0 / kvth0 * offset;
        *k2 += self.stk2 / kvth0.pow(self.lodk2) * offset;
        *eta0 += self.steta0 / kvth0.pow(self.lodeta0) * offset;
        Ok(())
    }

    /// Computes the five body-network conductances of `rbodymod` 1, scaled by `multiplier`.
    fn body_network(&self, multiplier: R) -> Option<[R; 5]> {
        match self.body_resistance_model {
            BodyResistance::Zero => None,
            BodyResistance::One => {
                let conductance = |resistance: R| {
                    if resistance.value() < 1.0e-3 {
                        R::constant(1.0e3)
                    } else {
                        resistance.recip() + self.gbmin
                    }
                };
                Some(
                    [self.rbpd, self.rbps, self.rbpb, self.rbdb, self.rbsb]
                        .map(|resistance| conductance(resistance) * multiplier),
                )
            }
        }
    }

    /// Computes the gate electrode's conductance (`b4v5temp.c`'s `grgeltd`, lines 1559 to
    /// 1568) and intrinsic-input resistance factors for `rgatemod` 1 to 3.
    fn gate_network(
        &self,
        binned: &Binned<R>,
        thermal: &Thermal<R>,
        geometry: &Geometry<R>,
        fingers: R,
    ) -> Option<Gate<R>> {
        (self.gate_resistance_model != GateResistance::Zero).then(|| {
            let ngcon = self.ngcon;
            let resistance = self.rshg * (self.xgw + geometry.weff_cj / 3.0 / ngcon)
                / (ngcon * fingers * (geometry.l_new - self.xgl));
            let electrode = if resistance.value() > 0.0 {
                resistance.recip()
            } else {
                R::constant(1.0e3)
            };
            Gate {
                model: self.gate_resistance_model,
                electrode,
                xrcrg1_nf: binned.xrcrg1 * fingers,
                xrcrg2_vtm: binned.xrcrg2 * thermal.vtm,
            }
        })
    }

    /// Returns `[source_area, drain_area, source_perimeter, drain_perimeter, gate_edge]`
    /// from given values or `b4v5geo.c`'s defaults.
    fn diffusion_areas(&self, geometry: &Geometry<R>, fingers: R) -> [R; 5] {
        let gate_contact_spacing = self.dmcg - self.dmcgt;
        let isolation_spacing = self.dmci.unwrap_or(self.dmcg);
        let [
            derived_source_perimeter,
            derived_drain_perimeter,
            derived_source_area,
            derived_drain_area,
        ] = diffusions(
            fingers.value(),
            geometry.weff_cj,
            [gate_contact_spacing, isolation_spacing],
        );
        let gate_edge = geometry.weff_cj * fingers;
        let perimeter = |given: Option<R>, derived: R| match (given, self.perimeter_model) {
            (Some(given), Perimeter::Zero) => given,
            (Some(given), Perimeter::One) => given - gate_edge,
            (None, _) => derived,
        };
        [
            self.source_area.unwrap_or(derived_source_area),
            self.drain_area.unwrap_or(derived_drain_area),
            perimeter(self.source_perimeter, derived_source_perimeter),
            perimeter(self.drain_perimeter, derived_drain_perimeter),
            gate_edge,
        ]
    }

    /// Builds `[drain_junction, source_junction]` for `diomod` 1 with trap-assisted
    /// tunneling.
    fn junctions(
        &self,
        thermal: &Thermal<R>,
        scalings: JunctionScalings<R>,
        [
            source_area,
            drain_area,
            source_perimeter,
            drain_perimeter,
            gate_edge,
        ]: [R; 5],
    ) -> Result<[Junction<R>; 2], ParameterError> {
        let (
            [source_densities, drain_densities],
            [source_emission, drain_emission],
            [source_threshold, drain_threshold],
        ) = scalings;
        let saturation = |area: R, perimeter: R, [bottom, sidewall, gate]: [R; 3]| {
            if area.value() <= 0.0 && perimeter.value() <= 0.0 {
                R::constant(1.0e-14)
            } else {
                area * bottom + perimeter * sidewall + gate_edge * gate
            }
        };
        let source_saturation = saturation(source_area, source_perimeter, source_densities);
        let drain_saturation = saturation(drain_area, drain_perimeter, drain_densities);
        let bottom_slope = thermal.vtm0 * (self.njts * (self.tnjts * thermal.rise + 1.0));
        let sidewall_slope = thermal.vtm0 * (self.njtssw * (self.tnjtssw * thermal.rise + 1.0));
        let gate_slope = thermal.vtm0 * (self.njtsswg * (self.tnjtsswg * thermal.rise + 1.0));
        let t7 = thermal.eg0 / thermal.vtm * thermal.rise;
        let tunneling = |[bottom_current, sidewall_current, gate_current]: [R; 3],
                         [bottom_exponent, sidewall_exponent, gate_exponent]: [R; 3],
                         [bottom_voltage, sidewall_voltage, gate_voltage]: [R; 3],
                         [area, perimeter]: [R; 2]| {
            [
                (
                    dexp(D::constant(bottom_exponent * t7)).value * area * bottom_current,
                    bottom_voltage,
                    bottom_slope,
                ),
                (
                    dexp(D::constant(sidewall_exponent * t7)).value * perimeter * sidewall_current,
                    sidewall_voltage,
                    sidewall_slope,
                ),
                (
                    dexp(D::constant(gate_exponent * t7)).value * gate_edge * gate_current,
                    gate_voltage,
                    gate_slope,
                ),
            ]
        };
        let source_vts = [self.vtss, self.vtssws, self.vtsswgs];
        let drain_vts = [
            self.vtsd.unwrap_or(self.vtss),
            self.vtsswd.unwrap_or(self.vtssws),
            self.vtsswgd.unwrap_or(self.vtsswgs),
        ];
        for (name, value) in ["vtss", "vtssws", "vtsswgs"]
            .into_iter()
            .zip(source_vts)
            .chain(["vtsd", "vtsswd", "vtsswgd"].into_iter().zip(drain_vts))
        {
            ParameterError::non_negative(name, value)?;
        }
        let source_junction = build_junction(
            thermal.vtm,
            source_saturation,
            source_emission,
            source_threshold,
            tunneling(
                [self.jtss, self.jtssws, self.jtsswgs],
                [self.xtss, self.xtssws, self.xtsswgs],
                source_vts,
                [source_area, source_perimeter],
            ),
        );
        let drain_junction = build_junction(
            thermal.vtm,
            drain_saturation,
            drain_emission,
            drain_threshold,
            tunneling(
                [
                    self.jtsd.unwrap_or(self.jtss),
                    self.jtsswd.unwrap_or(self.jtssws),
                    self.jtsswgd.unwrap_or(self.jtsswgs),
                ],
                [
                    self.xtsd.unwrap_or(self.xtss),
                    self.xtsswd.unwrap_or(self.xtssws),
                    self.xtsswgd.unwrap_or(self.xtsswgs),
                ],
                drain_vts,
                [drain_area, drain_perimeter],
            ),
        );
        Ok([drain_junction, source_junction])
    }

    /// Applies `paramchk = 1` clamps and builds `rdsmod` 1's external resistances,
    /// zeroing the internal channel resistance terms when `rdsmod` is 1.
    fn clamp_and_external(
        &self,
        core: &mut Core<R>,
        thermal: &Thermal<R>,
        pow_weff_wr: R,
        nsd: R,
    ) -> Option<External<R>> {
        if self.check_parameters {
            // ngspice's `paramchk = 1` clamps what it warns of.
            if core.binned.a2.value() < 0.01 {
                core.binned.a2 = R::constant(0.01);
            } else if core.binned.a2.value() > 1.0 {
                core.binned.a2 = R::constant(1.0);
                core.binned.a1 = R::constant(0.0);
            }
            if core.binned.prwg.value() < 0.0 {
                core.binned.prwg = R::constant(0.0);
            }
            if core.binned.rdsw.value() < 0.0 {
                core.binned.rdsw = R::constant(0.0);
                core.rds0 = R::constant(0.0);
            }
            if core.rds0.value() < 0.0 {
                core.rds0 = R::constant(0.0);
            }
            if core.rdswmin.value() < 0.0 {
                core.rdswmin = R::constant(0.0);
            }
        }
        let external =
            (self.drain_source_resistance_model == DrainSourceResistance::One).then(|| {
                let t10 = core.binned.prt * thermal.rise;
                let at_least_zero = |resistance: R| {
                    if resistance.value() < 0.0 {
                        R::constant(0.0)
                    } else {
                        resistance
                    }
                };
                let sheet = [self.drain_squares, self.source_squares].map(|squares| match squares
                    .map(|squares| self.rsh * squares)
                {
                    Some(resistance) if resistance.value() > 0.0 => resistance.recip(),
                    _ => R::constant(1.0e3),
                });
                External {
                    minimums: [self.rdwmin, self.rswmin]
                        .map(|minimum| at_least_zero(minimum + t10) / pow_weff_wr),
                    resistances: [core.binned.rdw, core.binned.rsw]
                        .map(|resistance| at_least_zero(resistance + t10) / pow_weff_wr),
                    vfbsd: if core.ngate.value() > 0.0 {
                        thermal.vtm0 * (core.ngate / nsd).ln()
                    } else {
                        R::constant(0.0)
                    },
                    prwg: core.binned.prwg,
                    prwb: core.binned.prwb,
                    sheet,
                }
            });
        if external.is_some() {
            // `b4v5ld.c` takes no resistance within the drain current at `rdsmod` 1.
            core.rds0 = R::constant(0.0);
            core.rdswmin = R::constant(0.0);
        }
        external
    }

    /// Computes gate-overlap capacitances and `ckappa` clamps from the oxide.
    fn overlaps(
        &self,
        binned: &Binned<R>,
        coxe: R,
        geometry: &Geometry<R>,
        fingers: R,
    ) -> Overlaps<R> {
        let [source_model, drain_model, bulk_model] = default_overlaps(
            [self.cgso, self.cgdo, self.cgbo],
            [self.cgsl, self.cgdl],
            (coxe, self.xj),
            (self.dlc, geometry.dwc_model),
        );
        let at_least_kappa = |kappa: R| {
            if kappa.value() < 0.02 {
                R::constant(0.02)
            } else {
                kappa
            }
        };
        Overlaps {
            cgso: (source_model + binned.cf) * geometry.weff_cv,
            cgdo: (drain_model + binned.cf) * geometry.weff_cv,
            cgsl: binned.cgsl * geometry.weff_cv,
            cgdl: binned.cgdl * geometry.weff_cv,
            ckappas: at_least_kappa(binned.ckappas),
            ckappad: at_least_kappa(binned.ckappad),
            cgbo: bulk_model * geometry.leff_cv * fingers,
            fingers,
        }
    }

    /// Computes `[drain_depletion, source_depletion]` junction capacitances and built-in
    /// potentials at `thermal`'s temperature.
    fn depletions(
        &self,
        thermal: &Thermal<R>,
        [
            source_area,
            drain_area,
            source_perimeter,
            drain_perimeter,
            gate_edge,
        ]: [R; 5],
    ) -> [Depletion<R>; 2] {
        let gate_capacitance = self.cjswgs.unwrap_or(self.cjsws);
        let gate_potential = self.pbswgs.unwrap_or(self.pbsws);
        let gate_grading = self.mjswgs.unwrap_or(self.mjsws);
        let build = |[bottom_cap, sidewall_cap, gate_cap]: [R; 3],
                     [bottom_pot, sidewall_pot, gate_pot]: [R; 3],
                     gradings: [R; 3],
                     [area, perimeter]: [R; 2]| Depletion {
            capacitances: [
                heated_capacitance(bottom_cap, self.tcj, thermal.delta) * area,
                heated_capacitance(sidewall_cap, self.tcjsw, thermal.delta) * perimeter,
                heated_capacitance(gate_cap, self.tcjswg, thermal.delta) * gate_edge,
            ],
            potentials: [
                heated_potential(bottom_pot, self.tpb, thermal.delta),
                heated_potential(sidewall_pot, self.tpbsw, thermal.delta),
                heated_potential(gate_pot, self.tpbswg, thermal.delta),
            ],
            gradings,
        };
        let source_depletion = build(
            [self.cjs, self.cjsws, gate_capacitance],
            [self.pbs, self.pbsws, gate_potential],
            [self.mjs, self.mjsws, gate_grading],
            [source_area, source_perimeter],
        );
        let drain_depletion = build(
            [
                self.cjd.unwrap_or(self.cjs),
                self.cjswd.unwrap_or(self.cjsws),
                self.cjswgd.unwrap_or(gate_capacitance),
            ],
            [
                self.pbd.unwrap_or(self.pbs),
                self.pbswd.unwrap_or(self.pbsws),
                self.pbswgd.unwrap_or(gate_potential),
            ],
            [
                self.mjd.unwrap_or(self.mjs),
                self.mjswd.unwrap_or(self.mjsws),
                self.mjswgd.unwrap_or(gate_grading),
            ],
            [drain_area, drain_perimeter],
        );
        [drain_depletion, source_depletion]
    }

    /// Resolves the flicker and thermal noise parameters for `geometry`.
    fn noise_parameters(&self, geometry: &Geometry<R>) -> Noise {
        Noise {
            flicker_model: self.flicker_noise_model,
            thermal_model: self.thermal_noise_model,
            noia: self
                .noia
                .map_or(by_channel(self.channel, 6.25e41, 6.188e40), Real::value),
            noib: self
                .noib
                .map_or(by_channel(self.channel, 3.125e26, 1.5e25), Real::value),
            noic: self.noic.value(),
            em: self.em.value(),
            ef: self.ef.value(),
            af: self.af.value(),
            kf: self.kf.value(),
            ntnoi: self.ntnoi.value(),
            tnoia: self.tnoia.value(),
            tnoib: self.tnoib.value(),
            rnoia: self.rnoia.value(),
            rnoib: self.rnoib.value(),
            length: (geometry.leff - self.lintnoi * 2.0).value(),
        }
    }

    /// Fails on what `b4v5check.c` stops on in the binned parameters at the effective
    /// length and width, the depletion's and the gate's dopings, the surface potential
    /// and the mobility and saturation velocity at the temperature.
    fn checked(
        &self,
        binned: &Binned<R>,
        [leff, weff]: [R; 2],
        [ndep, nsub, phi, ngate]: [R; 4],
        [u0temp, vsattemp]: [R; 2],
    ) -> Result<(), ParameterError> {
        let leff = leff.value();
        let weff = weff.value();
        for (name, value) in [("lpe0", binned.lpe0), ("lpeb", binned.lpeb)] {
            check(value.value() < -leff, || {
                format!(
                    "{name}, {:?}, is at least the negative of the effective length, {leff:?}",
                    value.value()
                )
            })?;
        }
        check(self.lintnoi.value() > leff / 2.0, || {
            format!(
                "lintnoi, {:?}, is at most half the effective length, {leff:?}",
                self.lintnoi.value()
            )
        })?;
        for (name, value) in [
            ("ndep", ndep),
            ("phi", phi),
            ("nsub", nsub),
            ("xj", binned.xj),
            ("u0 at the temperature", u0temp),
            ("vsat at the temperature", vsattemp),
            ("pclm", binned.pclm),
        ] {
            ParameterError::positive(name, value)?;
        }
        check(ngate.value() < 0.0 || ngate.value() > 1.0e25, || {
            format!(
                "ngate lies between 0 and 1e31 per cubic meter, not {:?}",
                binned.ngate.value()
            )
        })?;
        for (name, value) in [
            ("dvt1", binned.dvt1),
            ("dvt1w", binned.dvt1w),
            ("dsub", binned.dsub),
            ("delta", binned.delta),
            ("drout", binned.drout),
            ("fprout", binned.fprout),
            ("pdits", binned.pdits),
            ("pditsl", self.pditsl),
            ("clc", binned.clc),
        ] {
            ParameterError::non_negative(name, value)?;
        }
        for (name, value) in [("w0", binned.w0), ("b1", binned.b1)] {
            #[expect(
                clippy::float_cmp,
                reason = "b4v5check.c stops on this exact equality, a division by zero"
            )]
            let vanishing = value.value() == -weff;
            check(vanishing, || {
                format!("{name} differs from the negative of the effective width, {weff:?}")
            })?;
        }
        Ok(())
    }
}
