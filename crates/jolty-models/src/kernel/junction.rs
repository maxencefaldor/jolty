//! What every semiconductor junction shares: its current, its depletion charge, and how
//! Newton steps of its voltages are limited.
//!
//! The equations and the limiting follow SPICE, as ngspice implements them, so a junction
//! behaves the same in every model that has one: the diode, the JFET's gate and the
//! bipolar transistor's base.
//!
//! Derived from ngspice 47's implementations of junctions, in `src/spicelib/devices/dio/`
//! and `devsup.c`, their limiting among them:
//! Copyright 1985 - 2018, Regents of the University of California and others. Used under
//! ngspice's Modified BSD license, which the repository's `NOTICE` reproduces.

use crate::kernel::dual::Dual;
use crate::{ParameterError, Real};

/// The Boltzmann constant, in joules per kelvin.
pub const BOLTZMANN: f64 = 1.380_649e-23;
/// The elementary charge, in coulombs.
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
/// SPICE's reference temperature, 27 °C, in kelvins: the temperature a circuit is simulated
/// at, and its devices' parameters were measured at, unless said otherwise.
pub const REFERENCE_TEMPERATURE: f64 = 300.15;

/// The thermal voltage `k·T / q` at `temperature`, in volts.
#[must_use]
pub const fn thermal_voltage(temperature: f64) -> f64 {
    BOLTZMANN * temperature / ELEMENTARY_CHARGE
}

/// Silicon's band gap at `temperature`, in electronvolts, as SPICE models it.
pub(crate) fn band_gap<R: Real>(temperature: R) -> R {
    -(temperature * 7.02e-4 * temperature / (temperature + 1108.0) - 1.16)
}

/// The temperature a device is evaluated at, and the nominal temperature its parameters
/// were measured at, in kelvins.
///
/// It scales parameters from the nominal temperature as ngspice does: saturation currents
/// with the energy gap, built-in potentials and depletion capacitances with silicon's band
/// gap, and other parameters by their own coefficients. The device's temperature is a
/// number like any parameter, so a device that heats carries every scaled parameter's
/// derivative with respect to it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Temperatures<R: Real = f64> {
    /// The temperature the device is evaluated at.
    pub(crate) actual: R,
    /// The temperature its parameters were measured at.
    pub(crate) nominal: R,
}

impl<R: Real> Temperatures<R> {
    /// The device's temperature, `actual`, and its parameters' `nominal` temperature.
    ///
    /// # Errors
    ///
    /// Returns an error when the nominal temperature is not positive.
    pub(crate) fn new(actual: R, nominal: R) -> Result<Self, ParameterError> {
        ParameterError::positive("nominal_temperature", nominal)?;
        Ok(Self { actual, nominal })
    }

    /// A junction's temperatures, `actual` and `nominal`, both in kelvins, as
    /// [`Temperatures::new`] makes them: a junction at 0 K has no thermal voltage.
    ///
    /// # Errors
    ///
    /// Returns an error when either temperature is not positive.
    pub(crate) fn of_junction(actual: R, nominal: R) -> Result<Self, ParameterError> {
        if actual.value() <= 0.0 {
            return Err(ParameterError(format!(
                "a junction's temperature lies above 0 K, not at {:?} K",
                actual.value()
            )));
        }
        Self::new(actual, nominal)
    }

    /// The device's temperature less the nominal temperature.
    pub(crate) fn difference(self) -> R {
        -(self.nominal - self.actual)
    }

    /// The factor `1 + tc1 · ΔT + tc2 · ΔT²` of a parameter with linear coefficient
    /// `linear` and quadratic coefficient `quadratic`.
    pub(crate) fn polynomial(self, linear: R, quadratic: R) -> R {
        let difference = self.difference();
        (linear + quadratic * difference) * difference + 1.0
    }
}

impl<R: Real> Temperatures<R> {
    /// The thermal voltage at the device's temperature.
    pub(crate) fn thermal_voltage(self) -> R {
        self.actual * BOLTZMANN / ELEMENTARY_CHARGE
    }

    /// The device's temperature over the nominal temperature.
    pub(crate) fn ratio(self) -> R {
        self.actual / self.nominal
    }

    /// The factor that scales a saturation current with `emission` coefficient, given the
    /// `energy_gap` in volts and the temperature `exponent`, SPICE's XTI:
    /// `exp(((T/Tn − 1) · Eg / Vt + XTI · ln(T/Tn)) / n)`.
    pub(crate) fn saturation(self, energy_gap: R, exponent: R, emission: R) -> R {
        let ratio = self.ratio();
        (((ratio - 1.0) * energy_gap / self.thermal_voltage() + exponent * ratio.ln()) / emission)
            .exp()
    }

    /// A junction's built-in `potential` and zero-bias `capacitance`, of `grading`
    /// coefficient, scaled to the device's temperature.
    pub(crate) fn junction(self, potential: R, capacitance: R, grading: R) -> (R, R) {
        let nominal = self.nominal / REFERENCE_TEMPERATURE;
        let intrinsic = (potential - potential_shift(self.nominal)) / nominal;
        let scaled =
            intrinsic * (self.actual / REFERENCE_TEMPERATURE) + potential_shift(self.actual);
        let stretch = |temperature: R, potential: R| {
            grading
                * ((temperature - REFERENCE_TEMPERATURE) * 4e-4
                    - (potential - intrinsic) / intrinsic)
                + 1.0
        };
        (
            scaled,
            capacitance * stretch(self.actual, scaled) / stretch(self.nominal, potential),
        )
    }
}

/// How far a junction's built-in potential at `temperature` lies from the reference
/// temperature's, through silicon's band gap and intrinsic concentration: SPICE's `pbfact`.
fn potential_shift<R: Real>(temperature: R) -> R {
    let gap = -band_gap(temperature) / (temperature * (2.0 * BOLTZMANN))
        + 1.115_087_7 / (2.0 * BOLTZMANN * REFERENCE_TEMPERATURE);
    temperature * BOLTZMANN / ELEMENTARY_CHARGE
        * -2.0
        * ((temperature / REFERENCE_TEMPERATURE).ln() * 1.5 + gap * ELEMENTARY_CHARGE)
}

/// The current through a junction at `voltage`, with its derivatives, of `saturation`
/// current and exponential `slope` (the emission coefficient times the thermal voltage).
///
/// Forward and slightly reversed, the current is `Is · (e^{v / slope} − 1)`. Below three
/// slopes of reverse bias it follows SPICE's form, `−Is · (1 + (3·slope / (e·v))³)`, which
/// joins the exponential with its slope and settles on `−Is`.
pub(crate) fn junction_current<R: Real, const N: usize>(
    voltage: Dual<R, N>,
    saturation: R,
    slope: R,
) -> Dual<R, N> {
    if voltage.value.value() >= -(slope.value() * 3.0) {
        (voltage.divide(slope).exp() - 1.0).scale(saturation)
    } else {
        let ratio = Dual::constant(slope * 3.0) / (voltage * std::f64::consts::E);
        let cube = ratio * ratio * ratio;
        -(cube + 1.0).scale(saturation)
    }
}

/// The current through a junction at `voltage` and its derivative, as [`junction_current`]
/// computes them, for the kernels that still write their own derivatives by hand, the
/// JFET's and the bipolar transistor's.
pub(crate) fn junction_current_with_slope<R: Real>(
    voltage: f64,
    saturation: R,
    slope: R,
) -> (R, R) {
    let [voltage] = Dual::variables([voltage]);
    let Dual {
        value,
        gradient: [derivative],
    } = junction_current(voltage, saturation, slope);
    (value, derivative)
}

/// The voltage above which a junction's steps are limited: where its current's curvature
/// takes over, `slope · ln(slope / (√2 · Is))`.
pub(crate) fn critical_voltage(saturation: f64, slope: f64) -> f64 {
    slope * (slope / (std::f64::consts::SQRT_2 * saturation)).ln()
}

/// SPICE's limiting of a junction voltage's step from `old` to `new`, with exponential
/// `slope` and `critical` voltage. Above the critical voltage, a step of more than two
/// slopes forward becomes a logarithmic one the exponential follows safely, and one
/// backward falls to the critical voltage, as in SPICE3. A step deep into reverse bias is
/// bounded, as in ngspice. Returns the limited voltage, or `None` when `new` needs no
/// limiting.
#[must_use]
pub fn limit_junction(new: f64, old: f64, slope: f64, critical: f64) -> Option<f64> {
    if new > critical && (new - old).abs() > 2.0 * slope {
        Some(if old > 0.0 {
            let argument = 1.0 + (new - old) / slope;
            if argument > 0.0 {
                old + slope * argument.ln()
            } else {
                critical
            }
        } else {
            slope * (new / slope).ln()
        })
    } else if new < 0.0 {
        let floor = if old > 0.0 {
            -old - 1.0
        } else {
            2.0 * old - 1.0
        };
        (new < floor).then_some(floor)
    } else {
        None
    }
}

/// SPICE's limiting of a field-effect transistor's gate voltage's step from `old` to `new`,
/// around its `threshold`: steps are bounded relative to the distance from the threshold,
/// so the drain current's square law cannot overshoot. Returns the limited voltage, or
/// `None` when `new` needs no limiting.
#[must_use]
pub fn limit_gate(new: f64, old: f64, threshold: f64) -> Option<f64> {
    let high = (2.0 * (old - threshold)).abs() + 2.0;
    let low = (old - threshold).abs() + 1.0;
    let on = threshold + 3.5;
    let step = new - old;
    if old >= threshold {
        if old >= on {
            if step <= 0.0 {
                if new >= on {
                    (-step > low).then_some(old - low)
                } else {
                    (new < threshold + 2.0).then_some(threshold + 2.0)
                }
            } else {
                (step > high).then_some(old + high)
            }
        } else if step <= 0.0 {
            (new < threshold - 0.5).then_some(threshold - 0.5)
        } else {
            (new > threshold + 4.0).then_some(threshold + 4.0)
        }
    } else if step <= 0.0 {
        (-step > high).then_some(old - high)
    } else {
        let ceiling = threshold + 0.5;
        if new <= ceiling {
            (step > low).then_some(old + low)
        } else {
            Some(ceiling)
        }
    }
}

/// SPICE's limiting of a field-effect transistor's steps of its gate-source and
/// drain-source voltages from `old` to `new`, around its `threshold`: the gate's step from
/// whichever end acted as the source, then the drain's. Returns the voltages as limited, and
/// whether either step was.
pub(crate) fn limit_channel(new: [f64; 2], old: [f64; 2], threshold: f64) -> ([f64; 2], bool) {
    let [mut vgs, mut vds] = new;
    let [previous_gate, previous_drain] = old;
    let mut limited = false;
    let mut apply = |value: f64, limit: Option<f64>| {
        limited |= limit.is_some();
        limit.unwrap_or(value)
    };
    let vgd = vgs - vds;
    if previous_drain >= 0.0 {
        vgs = apply(vgs, limit_gate(vgs, previous_gate, threshold));
        vds = vgs - vgd;
        vds = apply(vds, limit_drain(vds, previous_drain));
    } else {
        let vgd = apply(
            vgd,
            limit_gate(vgd, previous_gate - previous_drain, threshold),
        );
        vds = vgs - vgd;
        vds = -apply(-vds, limit_drain(-vds, -previous_drain));
        vgs = vgd + vds;
    }
    ([vgs, vds], limited)
}

/// SPICE's limiting of a field-effect transistor's drain voltage's step from `old` to
/// `new`, both with the drain at or above the source. Returns the limited voltage, or
/// `None` when `new` needs no limiting.
pub(crate) fn limit_drain(new: f64, old: f64) -> Option<f64> {
    if old >= 3.5 {
        if new > old {
            (new > 3.0 * old + 2.0).then_some(3.0 * old + 2.0)
        } else {
            (new < 2.0).then_some(2.0)
        }
    } else if new > old {
        (new > 4.0).then_some(4.0)
    } else {
        (new < -0.5).then_some(-0.5)
    }
}

/// The depletion charge of a junction: the classic power law below `threshold`, extended
/// linearly in capacitance above it, where the power law would diverge.
///
/// With zero-bias capacitance `C`, built-in potential `φ`, grading coefficient `m` and
/// depletion coefficient `fc`, the charge is `C · φ · (1 − (1 − v / φ)^{1 − m}) / (1 − m)`
/// below `fc · φ`, and above it the charge there plus
/// `C · ((1 − fc · (1 + m)) · (v − fc · φ) + m / (2φ) · (v² − (fc · φ)²)) / (1 − fc)^{1 + m}`.
/// The factors that do not depend on `v` are prepared once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Depletion<R: Real = f64> {
    /// `1 / φ`.
    inverse_potential: R,
    /// `−m`.
    exponent: R,
    /// `C · φ / (1 − m)`.
    scale: R,
    /// `fc · φ`.
    threshold: R,
    /// The charge at the threshold.
    at_threshold: R,
    /// `C · (1 − fc · (1 + m)) / (1 − fc)^{1 + m}`.
    linear: R,
    /// `C · m / (2φ · (1 − fc)^{1 + m})`.
    quadratic: R,
}

impl<R: Real> Depletion<R> {
    /// The depletion of a junction with zero-bias `capacitance`, built-in `potential`,
    /// `grading` coefficient and depletion `coefficient`, the fraction of the potential
    /// above which the capacitance is extended linearly; `None` without capacitance.
    pub(crate) fn new(capacitance: R, potential: R, grading: R, coefficient: R) -> Option<Self> {
        (capacitance.value() > 0.0).then(|| {
            let remaining = -(coefficient - 1.0);
            let scale = capacitance * potential / -(grading - 1.0);
            let divisor = remaining.pow(grading + 1.0);
            Self {
                inverse_potential: potential.recip(),
                exponent: -grading,
                scale,
                threshold: coefficient * potential,
                at_threshold: scale * -(remaining.pow(-(grading - 1.0)) - 1.0),
                linear: capacitance * -(coefficient * (grading + 1.0) - 1.0) / divisor,
                quadratic: capacitance * grading / (potential * 2.0 * divisor),
            }
        })
    }

    /// The charge, with its derivatives, at junction voltage `voltage`.
    pub(crate) fn evaluate<const N: usize>(&self, voltage: Dual<R, N>) -> Dual<R, N> {
        let threshold = self.threshold;
        if voltage.value.value() < threshold.value() {
            let remaining = -(voltage.scale(self.inverse_potential) - 1.0);
            (-(remaining * remaining.pow(self.exponent) - 1.0)).scale(self.scale)
        } else {
            let above = voltage.plus(-threshold);
            let square = (voltage * voltage).plus(-(threshold * threshold));
            (above.scale(self.linear) + square.scale(self.quadratic)).plus(self.at_threshold)
        }
    }
}

/// Checks that a coefficient `value` is at least 0 and below 1.
///
/// # Errors
///
/// Returns an error naming `parameter` otherwise.
pub(crate) fn below_one<R: Real>(parameter: &str, value: R) -> Result<R, ParameterError> {
    if (0.0..1.0).contains(&value.value()) {
        Ok(value)
    } else {
        let value = value.value();
        Err(ParameterError(format!(
            "{parameter} must be at least 0 and below 1, not {value:?}"
        )))
    }
}

/// The charge, with its derivatives, of an optional depletion at `voltage`, zero without
/// one.
pub(crate) fn depletion_charge<R: Real, const N: usize>(
    depletion: Option<&Depletion<R>>,
    voltage: Dual<R, N>,
) -> Dual<R, N> {
    depletion.map_or(Dual::number(0.0), |depletion| depletion.evaluate(voltage))
}

/// The charge of an optional depletion at `voltage` and its capacitance, as
/// [`depletion_charge`] computes them, for the kernels that still write their own
/// derivatives by hand, the JFET's and the bipolar transistor's.
pub(crate) fn depletion_charge_with_capacitance<R: Real>(
    depletion: Option<&Depletion<R>>,
    voltage: f64,
) -> (R, R) {
    let [voltage] = Dual::variables([voltage]);
    let Dual {
        value,
        gradient: [capacitance],
    } = depletion_charge(depletion, voltage);
    (value, capacitance)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// The reverse-bias form joins the exponential with the same current and slope.
        #[test]
        fn the_junction_current_is_smooth_and_its_slope_is_its_derivative(
            voltage in -10.0..0.8f64,
        ) {
            let slope = 1.5 * thermal_voltage(REFERENCE_TEMPERATURE);
            let current = |voltage| junction_current(Dual::<f64, 1>::number(voltage), 1e-12, slope);
            let [at] = Dual::variables([voltage]);
            let Dual { value, gradient: [conductance] } = junction_current(at, 1e-12, slope);
            let step = 1e-7;
            let numeric = (current(voltage + step).value - current(voltage - step).value)
                / (2.0 * step);
            prop_assert!((numeric - conductance).abs() <= 1e-6 * conductance.abs() + 1e-18);
            prop_assert!(value >= -1e-12 * (1.0 + 1e-9));
        }

        /// A limited step is finite, and a forward junction step, any gate step, and any
        /// drain step at or above the source stay between the old voltage and the new one.
        #[test]
        fn limiting_never_overshoots(new in -20.0..20.0f64, old in -20.0..20.0f64) {
            let slope = thermal_voltage(REFERENCE_TEMPERATURE);
            let critical = critical_voltage(1e-14, slope);
            let between = |limited: f64| {
                limited.is_finite() && (limited - old).abs() <= (new - old).abs() + 1e-12
                    && (limited - old) * (new - old) >= 0.0
            };
            if let Some(limited) = limit_junction(new, old, slope, critical) {
                prop_assert!(limited.is_finite());
                prop_assert!(new < old || between(limited));
            }
            if let Some(limited) = limit_gate(new, old, -1.0) {
                prop_assert!(between(limited));
            }
            let (forward, previous) = (new.abs(), old.abs());
            if let Some(limited) = limit_drain(forward, previous) {
                prop_assert!((limited - previous).abs() <= (forward - previous).abs() + 1e-12);
                prop_assert!((limited - previous) * (forward - previous) >= 0.0);
            }
        }
    }

    #[test]
    fn the_reverse_form_meets_the_exponential_at_three_slopes() {
        let slope = thermal_voltage(REFERENCE_TEMPERATURE);
        let boundary = -3.0 * slope;
        let current = |voltage| {
            let [voltage] = Dual::<f64, 1>::variables([voltage]);
            let Dual {
                value,
                gradient: [slope],
            } = junction_current(voltage, 1e-14, slope);
            (value, slope)
        };
        let (exponential, exponential_slope) = current(boundary);
        let (cubic, cubic_slope) = current(boundary - 1e-15);
        assert!((exponential - cubic).abs() < 1e-24);
        assert!((exponential_slope / cubic_slope - 1.0).abs() < 1e-9);
    }
}
