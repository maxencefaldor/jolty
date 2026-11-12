//! The noise devices make: random excitations of the circuit's equations, each with its
//! one-sided power spectral density, as SPICE models them.

use crate::{BOLTZMANN, ELEMENTARY_CHARGE, Node, ParameterError, Real, Unknown};

/// A random excitation of the circuit's equations, whose one-sided power spectral density
/// is `white + flicker / f^exponent` at the frequency `f`: a current between two nodes, in
/// A²/Hz, or a voltage in series with a branch whose current is an unknown, in V²/Hz.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseSource {
    /// What makes the noise, such as `thermal` or `collector shot`.
    pub name: &'static str,
    /// The equations the excitation enters, positively and negatively: those of the nodes
    /// a current flows from and to, or a branch's own and none for a voltage in series
    /// with it.
    pub equations: [Node; 2],
    /// The density that does not change with frequency.
    pub white: f64,
    /// The flicker density at 1 Hz.
    pub flicker: f64,
    /// The power of the frequency the flicker density falls with.
    pub exponent: f64,
}

impl NoiseSource {
    /// The thermal noise current of a `conductance` between `nodes` at `temperature`, in
    /// kelvins: `4kT·g`.
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

    /// The thermal noise voltage of a `resistance` whose current is the unknown `branch`,
    /// at `temperature`, in kelvins: `4kT·r`, in series with the branch, so that a short
    /// makes none.
    #[must_use]
    pub fn thermal_in_series(
        name: &'static str,
        branch: Unknown,
        resistance: f64,
        temperature: f64,
    ) -> Self {
        Self::white(
            name,
            [Some(branch), None],
            4.0 * BOLTZMANN * temperature * resistance.abs(),
        )
    }

    /// The shot noise of a `current` crossing a junction: `2q·|I|`.
    #[must_use]
    pub fn shot(name: &'static str, nodes: [Node; 2], current: f64) -> Self {
        Self::white(name, nodes, 2.0 * ELEMENTARY_CHARGE * current.abs())
    }

    /// The flicker noise of a `current`: `coefficient · |I|^power / f`, as SPICE's `KF` and
    /// `AF` give it.
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

    /// A source of `density` at every frequency, entering `equations`.
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

    /// The density at `frequency`, in A²/Hz; the frequency must be positive where the
    /// source flickers.
    #[must_use]
    pub fn density(&self, frequency: f64) -> f64 {
        if self.flicker == 0.0 {
            self.white
        } else {
            self.white + self.flicker / frequency.powf(self.exponent)
        }
    }
}

/// SPICE's flicker noise of a device's current: its coefficient, `KF`, and the power of
/// the current, `AF`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Flicker {
    coefficient: f64,
    power: f64,
}

impl Flicker {
    /// The flicker noise of `coefficient` and `power`, checked as SPICE reads them.
    pub(crate) fn new<R: Real>(coefficient: R, power: R) -> Result<Self, ParameterError> {
        ParameterError::non_negative("flicker_coefficient", coefficient)?;
        ParameterError::positive("flicker_exponent", power)?;
        Ok(Self {
            coefficient: coefficient.value(),
            power: power.value(),
        })
    }

    /// The same noise, its coefficient multiplied by `factor`.
    pub(crate) fn scaled(self, factor: f64) -> Self {
        Self {
            coefficient: self.coefficient * factor,
            ..self
        }
    }

    /// Appends the flicker noise of `current` between `nodes` to `sources`, unless the
    /// coefficient is zero.
    pub(crate) fn add(self, nodes: [Node; 2], current: f64, sources: &mut Vec<NoiseSource>) {
        if self.coefficient > 0.0 {
            sources.push(NoiseSource::flicker(
                "flicker",
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
        let nodes = [Some(Unknown::new(0)), None];
        let thermal = NoiseSource::thermal("thermal", nodes, 1e-3, 300.0);
        assert!((thermal.density(1.0) - 4.0 * BOLTZMANN * 300.0 * 1e-3).abs() < 1e-35);
        let shot = NoiseSource::shot("shot", nodes, -1e-3);
        assert!((shot.density(1e6) - 2.0 * ELEMENTARY_CHARGE * 1e-3).abs() < 1e-35);
        let series = NoiseSource::thermal_in_series("thermal", Unknown::new(1), 50.0, 300.0);
        assert_eq!(series.equations, [Some(Unknown::new(1)), None]);
        assert!((series.density(1.0) - 4.0 * BOLTZMANN * 300.0 * 50.0).abs() < 1e-30);
        let flicker = NoiseSource::flicker("flicker", nodes, 1e-16, 2.0, 1e-3);
        assert!((flicker.density(10.0) - 1e-23).abs() < 1e-36);
        assert!((flicker.density(100.0) - 1e-24).abs() < 1e-37);
    }
}
