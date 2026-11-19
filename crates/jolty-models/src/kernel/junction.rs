//! Shared semiconductor junction physics: current-voltage equations, depletion charge,
//! temperature scaling, and Newton step limiting.
//!
//! Equations and limiting follow SPICE as implemented in ngspice, giving identical
//! junction behavior across diodes, JFET gates, and BJT bases.
//!
//! Derived from ngspice 47 junction implementations in `src/spicelib/devices/dio/` and
//! `devsup.c`: Copyright 1985 - 2018, Regents of the University of California and others.
//! Used under the ngspice Modified BSD license, reproduced in the repository `NOTICE` file.

use crate::kernel::dual::Dual;
use crate::{ParameterError, Real};

/// Boltzmann constant, in joules per kelvin.
pub const BOLTZMANN: f64 = 1.380_649e-23;
/// Elementary charge, in coulombs.
pub const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;
/// SPICE reference temperature (27 °C), in kelvins, used as the default simulation and
/// parameter measurement temperature.
pub const REFERENCE_TEMPERATURE: f64 = 300.15;

/// Computes the thermal voltage `k·T / q` at `temperature`, in volts.
#[must_use]
pub const fn thermal_voltage(temperature: f64) -> f64 {
    BOLTZMANN * temperature / ELEMENTARY_CHARGE
}

/// Computes the silicon band gap at `temperature`, in electronvolts, using the SPICE model.
pub(crate) fn band_gap<R: Real>(temperature: R) -> R {
    Varshni::silicon().at(temperature)
}

/// Band gap temperature dependence following Varshni's law,
/// `Eg(T) = Eg(0) − α·T² / (T + β)`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Varshni<R: Real = f64> {
    /// Band gap at 0 K, `Eg(0)`, in electronvolts.
    pub(crate) gap: R,
    /// Varshni linear parameter `α`, in electronvolts per kelvin.
    pub(crate) alpha: R,
    /// Varshni offset temperature `β`, in kelvins.
    pub(crate) beta: R,
}

impl<R: Real> Varshni<R> {
    /// Returns the SPICE silicon band-gap parameters (`1.16` eV at 0 K, `α = 7.02e-4` eV/K,
    /// `β = 1108` K, matching ngspice `diotemp.c`).
    pub(crate) fn silicon() -> Self {
        Self {
            gap: R::constant(1.16),
            alpha: R::constant(7.02e-4),
            beta: R::constant(1108.0),
        }
    }

    /// Evaluates the band gap at `temperature`, in kelvins.
    pub(crate) fn at(self, temperature: R) -> R {
        -(temperature * self.alpha * temperature / (temperature + self.beta) - self.gap)
    }
}

/// Actual device temperature and nominal parameter measurement temperature, in kelvins.
///
/// Scales parameters from nominal temperature following ngspice conventions: saturation
/// currents with the energy gap, built-in potentials and depletion capacitances with the
/// silicon band gap, and remaining parameters via dedicated coefficients. Representing
/// device temperature as a scalar `R` propagates temperature derivatives through every
/// scaled parameter during self-heating.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Temperatures<R: Real = f64> {
    /// Device operating temperature, in kelvins.
    pub(crate) actual: R,
    /// Nominal parameter measurement temperature, in kelvins.
    pub(crate) nominal: R,
}

impl<R: Real> Temperatures<R> {
    /// Constructs a temperature pair from `actual` device temperature and `nominal`
    /// measurement temperature.
    ///
    /// # Errors
    ///
    /// Returns an error when the nominal temperature is not positive.
    pub(crate) fn new(actual: R, nominal: R) -> Result<Self, ParameterError> {
        ParameterError::positive("nominal_temperature", nominal)?;
        Ok(Self { actual, nominal })
    }

    /// Constructs junction temperatures from `actual` and `nominal`, both in kelvins,
    /// enforcing positivity via [`Temperatures::new`] because a junction at 0 K lacks a
    /// nonzero thermal voltage.
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

    /// Returns the temperature offset `actual − nominal`.
    pub(crate) fn difference(self) -> R {
        -(self.nominal - self.actual)
    }

    /// Evaluates the quadratic temperature scaling factor `1 + tc1 · ΔT + tc2 · ΔT²` from
    /// `linear` and `quadratic` coefficients.
    pub(crate) fn polynomial(self, linear: R, quadratic: R) -> R {
        let difference = self.difference();
        (linear + quadratic * difference) * difference + 1.0
    }
}

impl<R: Real> Temperatures<R> {
    /// Computes the thermal voltage at the device's actual temperature.
    pub(crate) fn thermal_voltage(self) -> R {
        self.actual * BOLTZMANN / ELEMENTARY_CHARGE
    }

    /// Returns the ratio `actual / nominal`.
    pub(crate) fn ratio(self) -> R {
        self.actual / self.nominal
    }

    /// Computes the saturation current scaling factor
    /// `exp(((T/Tn − 1) · Eg / Vt + XTI · ln(T/Tn)) / n)` from `energy_gap` in volts,
    /// temperature `exponent` (SPICE `XTI`), and `emission` coefficient.
    pub(crate) fn saturation(self, energy_gap: R, exponent: R, emission: R) -> R {
        let ratio = self.ratio();
        (((ratio - 1.0) * energy_gap / self.thermal_voltage() + exponent * ratio.ln()) / emission)
            .exp()
    }

    /// Scales a junction's built-in `potential` and zero-bias `capacitance` of `grading`
    /// coefficient to the device's actual temperature using the silicon band gap.
    pub(crate) fn junction(self, potential: R, capacitance: R, grading: R) -> (R, R) {
        self.junction_of(potential, capacitance, grading, Varshni::silicon())
    }

    /// Scales a junction's built-in `potential` and zero-bias `capacitance` of `grading`
    /// coefficient to the device's actual temperature using band `gap`, matching ngspice
    /// `diotemp.c`.
    pub(crate) fn junction_of(
        self,
        potential: R,
        capacitance: R,
        grading: R,
        gap: Varshni<R>,
    ) -> (R, R) {
        let shift = |temperature: R| potential_shift(temperature, gap.at(temperature));
        let nominal = self.nominal / REFERENCE_TEMPERATURE;
        let intrinsic = (potential - shift(self.nominal)) / nominal;
        let scaled = intrinsic * (self.actual / REFERENCE_TEMPERATURE) + shift(self.actual);
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

/// Computes SPICE `pbfact`, the built-in potential shift at `temperature` relative to the
/// reference temperature given `band_gap` in electronvolts.
fn potential_shift<R: Real>(temperature: R, band_gap: R) -> R {
    let gap = -band_gap / (temperature * (2.0 * BOLTZMANN))
        + 1.115_087_7 / (2.0 * BOLTZMANN * REFERENCE_TEMPERATURE);
    temperature * BOLTZMANN / ELEMENTARY_CHARGE
        * -2.0
        * ((temperature / REFERENCE_TEMPERATURE).ln() * 1.5 + gap * ELEMENTARY_CHARGE)
}

/// Evaluates junction current and derivatives at `voltage` for `saturation` current and
/// exponential `slope` (emission coefficient times thermal voltage).
///
/// Forward and mild reverse bias use `Is · (e^{v / slope} − 1)`. Below `-3 · slope`,
/// switches to the SPICE cubic form `−Is · (1 + (3·slope / (e·v))³)`, matching value and
/// slope at the join and asymptoting to `−Is`.
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

/// Computes the critical voltage `slope · ln(slope / (√2 · Is))` above which junction
/// Newton steps undergo logarithmic limiting.
pub(crate) fn critical_voltage(saturation: f64, slope: f64) -> f64 {
    slope * (slope / (std::f64::consts::SQRT_2 * saturation)).ln()
}

/// Applies SPICE junction voltage limiting to a step from `old` to `new` given exponential
/// `slope` and `critical` voltage.
///
/// Above `critical`, forward steps exceeding `2 · slope` are compressed logarithmically and
/// backward steps fall back to `critical` (SPICE3). Deep reverse-bias steps are bounded as
/// in ngspice. Returns the limited voltage, or `None` when `new` requires no limiting.
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

/// Applies SPICE gate voltage limiting to a FET step from `old` to `new` around `threshold`.
///
/// Bounds steps relative to the distance from `threshold` to prevent quadratic drain-current
/// overshoot. Returns the limited voltage, or `None` when `new` requires no limiting.
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

/// Applies SPICE channel limiting to FET gate-source and drain-source voltage steps from
/// `old` to `new` around `threshold`.
///
/// Limits gate voltage relative to the active source end first, then limits drain-source
/// voltage. Returns the limited `[vgs, vds]` pair and a boolean indicating whether limiting
/// modified either step.
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

/// Applies SPICE drain voltage limiting to a FET step from `old` to `new` with the drain at
/// or above the source.
///
/// Returns the limited voltage, or `None` when `new` requires no limiting.
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

/// Junction depletion charge model combining a power law below `threshold` with a linear
/// capacitance extension above `threshold` to prevent divergence.
///
/// Given zero-bias capacitance `C`, built-in potential `φ`, grading coefficient `m`, and
/// depletion coefficient `fc`, charge equals `C·φ·(1 − (1 − v/φ)^{1 − m}) / (1 − m)` below
/// `fc·φ`. Above `fc·φ`, charge adds the quadratic integral of the linear capacitance:
/// `C·((1 − fc·(1 + m))·(v − fc·φ) + m/(2φ)·(v² − (fc·φ)²)) / (1 − fc)^{1 + m}`.
/// Voltage-independent factors are precomputed once during construction.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Depletion<R: Real = f64> {
    /// Reciprocal built-in potential `1 / φ`.
    inverse_potential: R,
    /// Negated grading coefficient `−m`.
    exponent: R,
    /// Power-law prefactor `C · φ / (1 − m)`.
    scale: R,
    /// Linear-extension voltage threshold `fc · φ`.
    threshold: R,
    /// Depletion charge evaluated at `threshold`.
    at_threshold: R,
    /// Linear capacitance coefficient `C · (1 − fc · (1 + m)) / (1 − fc)^{1 + m}`.
    linear: R,
    /// Quadratic charge coefficient `C · m / (2φ · (1 − fc)^{1 + m})`.
    quadratic: R,
}

impl<R: Real> Depletion<R> {
    /// Prepares a depletion charge model from zero-bias `capacitance`, built-in `potential`,
    /// `grading` coefficient, and linear-extension threshold `coefficient`, returning `None`
    /// when `capacitance` is zero.
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

    /// Evaluates depletion charge and derivatives at junction `voltage`.
    pub(crate) fn evaluate<const N: usize>(&self, voltage: Dual<R, N>) -> Dual<R, N> {
        let threshold = self.threshold;
        if voltage.value.value() < threshold.value() {
            // Evaluates `1 − (1 − v/φ)^{1 − m}` via `−expm1((1 − m) · ln1p(−v/φ))` to retain
            // precision near 0 V, where subtracting from 1 causes cancellation. At 1 nF, raw
            // error `ε·C·φ/(1 − m) ≈ 3.5e-25 C` scales by a 1.5e13 /s step weight to 5e-12 A,
            // exceeding the 1e-12 A nodal tolerance.
            let logarithm = (-voltage.scale(self.inverse_potential)).ln_1p();
            (-logarithm.scale(self.exponent + 1.0).exp_m1()).scale(self.scale)
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
            "{parameter} is at least 0 and below 1, not {value:?}"
        )))
    }
}

/// Evaluates depletion charge and derivatives at `voltage` for an optional [`Depletion`],
/// returning zero when `depletion` is `None`.
pub(crate) fn depletion_charge<R: Real, const N: usize>(
    depletion: Option<&Depletion<R>>,
    voltage: Dual<R, N>,
) -> Dual<R, N> {
    depletion.map_or(Dual::number(0.0), |depletion| depletion.evaluate(voltage))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// Verifies that the reverse-bias cubic joins the exponential with matching current
        /// and slope.
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

        /// Verifies that limited steps stay finite and that forward junction steps, gate
        /// steps, and forward drain steps remain between old and new voltages.
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
