//! Random noise excitations contributed by devices to the circuit equations, specified by
//! one-sided power spectral density following SPICE conventions.

use crate::{BOLTZMANN, ELEMENTARY_CHARGE, Node, ParameterError, Real, UnknownIndex};

/// Random circuit excitation with one-sided power spectral density
/// `white + flicker / f^exponent` at frequency `f`.
///
/// Represents either a noise current between two nodes (A²/Hz) or a series noise voltage on
/// a branch with an unknown current (V²/Hz).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseSource {
    /// Physical mechanism producing the noise, such as `thermal` or `collector shot`.
    pub name: &'static str,
    /// Equation pair receiving the positive and negative excitation stamps: two node
    /// equations for a current source, or `[Some(branch), None]` for a series voltage source.
    pub equations: [Node; 2],
    /// Frequency-independent white noise spectral density.
    pub white: f64,
    /// Flicker noise spectral density at 1 Hz.
    pub flicker: f64,
    /// Frequency roll-off exponent of the flicker noise component.
    pub exponent: f64,
}

impl NoiseSource {
    /// Creates a thermal noise current source `4kT·g` for `conductance` across `nodes` at
    /// `temperature` in kelvins.
    #[must_use]
    pub fn thermal(
        name: &'static str,
        nodes: [Node; 2],
        conductance: f64,
        temperature: f64,
    ) -> Self {
        Self::white(
            name,
            nodes,
            4.0 * BOLTZMANN * temperature * conductance.abs(),
        )
    }

    /// Creates a series thermal noise voltage source `4kT·r` on `branch` for `resistance` at
    /// `temperature` in kelvins, producing zero noise for a short circuit.
    #[must_use]
    pub fn thermal_in_series(
        name: &'static str,
        branch: UnknownIndex,
        resistance: f64,
        temperature: f64,
    ) -> Self {
        Self::white(
            name,
            [Some(branch), None],
            4.0 * BOLTZMANN * temperature * resistance.abs(),
        )
    }

    /// Creates a junction shot noise current source `2q·|I|` for `current` across `nodes`.
    #[must_use]
    pub fn shot(name: &'static str, nodes: [Node; 2], current: f64) -> Self {
        Self::white(name, nodes, 2.0 * ELEMENTARY_CHARGE * current.abs())
    }

    /// Creates a SPICE `KF`/`AF` flicker noise current source `coefficient · |I|^power / f`
    /// across `nodes`.
    #[must_use]
    pub fn flicker(
        name: &'static str,
        nodes: [Node; 2],
        coefficient: f64,
        power: f64,
        current: f64,
    ) -> Self {
        Self {
            name,
            equations: nodes,
            white: 0.0,
            flicker: coefficient * current.abs().powf(power),
            exponent: 1.0,
        }
    }

    /// Creates a white noise source with constant spectral `density` stamped into `equations`.
    #[must_use]
    pub fn white(name: &'static str, equations: [Node; 2], density: f64) -> Self {
        Self {
            name,
            equations,
            white: density,
            flicker: 0.0,
            exponent: 1.0,
        }
    }

    /// Evaluates the one-sided power spectral density at `frequency` (which is positive for
    /// sources with nonzero flicker).
    #[must_use]
    pub fn density(&self, frequency: f64) -> f64 {
        if self.flicker == 0.0 {
            self.white
        } else {
            self.white + self.flicker / frequency.powf(self.exponent)
        }
    }
}

/// SPICE current flicker noise parameters: coefficient `KF` and current exponent `AF`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Flicker {
    coefficient: f64,
    power: f64,
}

impl Flicker {
    /// Validates and constructs flicker noise parameters from `coefficient` (`KF`) and
    /// `power` (`AF`).
    pub(crate) fn new<R: Real>(coefficient: R, power: R) -> Result<Self, ParameterError> {
        ParameterError::non_negative("flicker_coefficient", coefficient)?;
        ParameterError::positive("flicker_exponent", power)?;
        Ok(Self {
            coefficient: coefficient.value(),
            power: power.value(),
        })
    }

    /// Returns a copy with the flicker coefficient multiplied by `factor`.
    pub(crate) fn scaled(self, factor: f64) -> Self {
        Self {
            coefficient: self.coefficient * factor,
            ..self
        }
    }

    /// Appends the flicker noise of `current` between `nodes` to `sources`, unless the
    /// coefficient is zero.
    pub(crate) fn add(self, nodes: [Node; 2], current: f64, sources: &mut Vec<NoiseSource>) {
        self.add_named("flicker", nodes, current, sources);
    }

    /// Appends the flicker noise of `current` between `nodes`, named `name`, to `sources`,
    /// unless the coefficient is zero.
    pub(crate) fn add_named(
        self,
        name: &'static str,
        nodes: [Node; 2],
        current: f64,
        sources: &mut Vec<NoiseSource>,
    ) {
        if self.coefficient > 0.0 {
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
}
