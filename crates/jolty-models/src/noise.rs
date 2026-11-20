//! Random noise excitations contributed by devices to the circuit equations, specified by
//! one-sided power spectral density following SPICE conventions.

use crate::{BOLTZMANN, ELEMENTARY_CHARGE, Node, ParameterError, Real, UnknownIndex};

/// Returns `|value|` with zero derivative at `0.0` so non-smooth noise laws remain finite
/// and symmetric at zero bias.
#[inline]
pub(crate) fn magnitude<R: Real>(value: R) -> R {
    if value.value() > 0.0 {
        value
    } else if value.value() < 0.0 {
        -value
    } else {
        R::constant(0.0)
    }
}

/// Returns `true` when `value` is positive or carries a nonzero seed derivative, preserving
/// right-hand sensitivity at zero without adding zero-valued sources to plain `f64` runs.
#[inline]
pub(crate) fn is_active<R: Real>(value: R) -> bool {
    value.value() > 0.0 || (0..R::SEEDS).any(|slot| value.derivative(slot) != 0.0)
}

/// Returns `true` when `value` and all of its seed derivatives vanish.
#[inline]
fn is_zero<R: Real>(value: R) -> bool {
    value.value() == 0.0 && (0..R::SEEDS).all(|slot| value.derivative(slot) == 0.0)
}

/// Random circuit excitation with one-sided power spectral density
/// `white + flicker / f^exponent` at frequency `f`.
///
/// Represents either a noise current between two nodes (A²/Hz) or a series noise voltage on
/// a branch with an unknown current (V²/Hz). Parameterized over scalar [`Real`] so the same
/// device equations yield both primal `f64` spectra and exact forward-mode derivatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseSource<R: Real = f64> {
    /// Physical mechanism producing the noise, such as `thermal` or `collector shot`.
    pub name: &'static str,
    /// Equation pair receiving the positive and negative excitation stamps: two node
    /// equations for a current source, or `[Some(branch), None]` for a series voltage source.
    pub equations: [Node; 2],
    /// Frequency-independent white noise spectral density.
    pub white: R,
    /// Flicker noise spectral density at 1 Hz.
    pub flicker: R,
    /// Frequency roll-off exponent of the flicker noise component.
    pub exponent: R,
}

impl<R: Real> NoiseSource<R> {
    /// Creates a thermal noise current source `4kT·g` for `conductance` across `nodes` at
    /// `temperature` in kelvins.
    #[must_use]
    pub fn thermal(name: &'static str, nodes: [Node; 2], conductance: R, temperature: R) -> Self {
        Self::white(
            name,
            nodes,
            R::constant(4.0 * BOLTZMANN) * temperature * magnitude(conductance),
        )
    }

    /// Creates a series thermal noise voltage source `4kT·r` on `branch` for `resistance` at
    /// `temperature` in kelvins, producing zero noise for a short circuit.
    #[must_use]
    pub fn thermal_in_series(
        name: &'static str,
        branch: UnknownIndex,
        resistance: R,
        temperature: R,
    ) -> Self {
        Self::white(
            name,
            [Some(branch), None],
            R::constant(4.0 * BOLTZMANN) * temperature * magnitude(resistance),
        )
    }

    /// Creates a junction shot noise current source `2q·|I|` for `current` across `nodes`.
    #[must_use]
    pub fn shot(name: &'static str, nodes: [Node; 2], current: R) -> Self {
        Self::white(
            name,
            nodes,
            R::constant(2.0 * ELEMENTARY_CHARGE) * magnitude(current),
        )
    }

    /// Creates a SPICE `KF`/`AF` flicker noise current source `coefficient · |I|^power / f`
    /// across `nodes`, returning finite zero derivatives at zero current.
    #[must_use]
    pub fn flicker(
        name: &'static str,
        nodes: [Node; 2],
        coefficient: R,
        power: R,
        current: R,
    ) -> Self {
        let base = magnitude(current);
        let density = if base.value() == 0.0 {
            if power.value() == 0.0 {
                coefficient
            } else {
                R::constant(0.0)
            }
        } else {
            coefficient * base.pow(power)
        };
        Self {
            name,
            equations: nodes,
            white: R::constant(0.0),
            flicker: density,
            exponent: R::constant(1.0),
        }
    }

    /// Creates a white noise source with constant spectral `density` stamped into `equations`.
    #[must_use]
    pub fn white(name: &'static str, equations: [Node; 2], density: R) -> Self {
        Self {
            name,
            equations,
            white: density,
            flicker: R::constant(0.0),
            exponent: R::constant(1.0),
        }
    }

    /// Evaluates the one-sided power spectral density at `frequency` (which is positive for
    /// sources with nonzero flicker).
    #[must_use]
    pub fn density(&self, frequency: f64) -> R {
        if is_zero(self.flicker) {
            self.white
        } else {
            self.white + self.flicker / R::constant(frequency).pow(self.exponent)
        }
    }

    /// Maps each spectral scalar of this noise source through `transform`.
    #[must_use]
    pub fn map<T: Real>(self, mut transform: impl FnMut(R) -> T) -> NoiseSource<T> {
        NoiseSource {
            name: self.name,
            equations: self.equations,
            white: transform(self.white),
            flicker: transform(self.flicker),
            exponent: transform(self.exponent),
        }
    }
}

/// SPICE current flicker noise parameters: coefficient `KF` and current exponent `AF`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Flicker<R: Real = f64> {
    coefficient: R,
    power: R,
}

impl<R: Real> Flicker<R> {
    /// Validates and constructs flicker noise parameters from `coefficient` (`KF`) and
    /// `power` (`AF`).
    pub(crate) fn new(coefficient: R, power: R) -> Result<Self, ParameterError> {
        ParameterError::non_negative("flicker_coefficient", coefficient)?;
        ParameterError::positive("flicker_exponent", power)?;
        Ok(Self { coefficient, power })
    }

    /// Returns a copy with the flicker coefficient multiplied by `factor`.
    pub(crate) fn scaled(self, factor: R) -> Self {
        Self {
            coefficient: self.coefficient * factor,
            ..self
        }
    }

    /// Appends the flicker noise of `current` between `nodes` to `sources`, unless the
    /// coefficient is inactive.
    pub(crate) fn add(self, nodes: [Node; 2], current: R, sources: &mut Vec<NoiseSource<R>>) {
        self.add_named("flicker", nodes, current, sources);
    }

    /// Appends the flicker noise of `current` between `nodes`, named `name`, to `sources`,
    /// unless the coefficient is inactive.
    pub(crate) fn add_named(
        self,
        name: &'static str,
        nodes: [Node; 2],
        current: R,
        sources: &mut Vec<NoiseSource<R>>,
    ) {
        if is_active(self.coefficient) {
            sources.push(NoiseSource::flicker(
                name,
                nodes,
                self.coefficient,
                self.power,
                current,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gradient;

    #[test]
    fn densities_follow_their_laws() {
        let nodes = [Some(UnknownIndex::new(0)), None];
        let thermal = NoiseSource::thermal("thermal", nodes, 1e-3, 300.0);
        assert!((thermal.density(1.0) - 4.0 * BOLTZMANN * 300.0 * 1e-3).abs() < 1e-35);
        let shot = NoiseSource::shot("shot", nodes, -1e-3);
        assert!((shot.density(1e6) - 2.0 * ELEMENTARY_CHARGE * 1e-3).abs() < 1e-35);
        let series = NoiseSource::thermal_in_series("thermal", UnknownIndex::new(1), 50.0, 300.0);
        assert_eq!(series.equations, [Some(UnknownIndex::new(1)), None]);
        assert!((series.density(1.0) - 4.0 * BOLTZMANN * 300.0 * 50.0).abs() < 1e-30);
        let flicker = NoiseSource::flicker("flicker", nodes, 1e-16, 2.0, 1e-3);
        assert!((flicker.density(10.0) - 1e-23).abs() < 1e-36);
        assert!((flicker.density(100.0) - 1e-24).abs() < 1e-37);
    }

    #[test]
    fn zero_current_noise_derivatives_are_finite_and_zero() {
        let nodes = [Some(UnknownIndex::new(0)), None];
        let zero_current = Gradient::new(0.0, [1.0, 0.0, 0.0]);
        let shot = NoiseSource::shot("shot", nodes, zero_current);
        assert_eq!(shot.white.value, 0.0);
        assert_eq!(shot.white.derivatives, [0.0, 0.0, 0.0]);

        for exponent in [0.5, 1.0, 1.5] {
            let coefficient = Gradient::new(1e-16, [0.0, 1.0, 0.0]);
            let power = Gradient::new(exponent, [0.0, 0.0, 1.0]);
            let flicker = NoiseSource::flicker("flicker", nodes, coefficient, power, zero_current);
            let density = flicker.density(100.0);
            assert_eq!(density.value, 0.0);
            assert!(density.derivatives.iter().all(|&d| d == 0.0));
        }
    }

    #[test]
    fn density_differentiates_flicker_and_exponent() {
        let nodes = [Some(UnknownIndex::new(0)), None];
        let source = NoiseSource {
            name: "flicker",
            equations: nodes,
            white: Gradient::new(2.0, [1.0, 0.0, 0.0]),
            flicker: Gradient::new(3.0, [0.0, 1.0, 0.0]),
            exponent: Gradient::new(1.5, [0.0, 0.0, 1.0]),
        };
        let frequency = 100.0_f64;
        let density = source.density(frequency);
        let expected_value = 2.0 + 3.0 / frequency.powf(1.5);
        let expected_flicker = frequency.powf(-1.5);
        let expected_exponent = -3.0 * frequency.powf(-1.5) * frequency.ln();
        // Tolerance of 1e-12 relative covers chain-rule exp/ln rounding in f64.
        assert!((density.value - expected_value).abs() < 1e-12);
        assert!((density.derivatives[0] - 1.0).abs() < 1e-12);
        assert!((density.derivatives[1] - expected_flicker).abs() < 1e-12);
        assert!((density.derivatives[2] - expected_exponent).abs() < 1e-12);
    }
}
