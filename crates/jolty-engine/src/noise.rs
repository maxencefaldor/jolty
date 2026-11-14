//! The noise at an output, from every device's noise sources, by the adjoint method.
//!
//! Linearized at its operating point, the circuit at a frequency is `(G + jωC) x = −b` for
//! an excitation `b` of its equations. One solve of the transposed system,
//! `(G + jωC)ᵀ z = e`, where `e` reads the output, 1 at its unknown and −1 at its
//! reference's, weighs every equation's effect on the output at once: an excitation `b`
//! moves the output by `−zᵀ b`. A noise source excites its two equations by +1 and −1, so
//! its density reaches the output times `|z₊ − z₋|²`, and the sources are independent, so
//! their densities add. The same weights give the gain from an input source's level, and
//! the output's noise over the gain's square is the noise referred to that input.

use jolty_models::{Input, Node, NoiseSource, Quantity};
use jolty_numerics::Complex64;
use serde::Deserialize;

use crate::ac::Linearization;
use crate::{OperatingPoint, Options, SimulationError, Simulator};

/// A noise analysis: the noise density at an output, at each frequency, and how much each
/// device's noise sources add to it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Noise {
    /// The output, an unknown by name, such as `out` or `V1:current`, or the reference
    /// node's name.
    pub output: String,
    /// The node the output voltage is taken over, for a differential output; by default
    /// the reference node.
    #[serde(default)]
    pub reference: Option<String>,
    /// The path of an independent source to refer the noise back to, such as `V1`.
    #[serde(default)]
    pub source: Option<String>,
    /// The frequencies, in hertz; positive.
    pub frequencies: Vec<f64>,
}

/// The noise at an output, at each frequency.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseSpectrum {
    /// The output's unknown, by name.
    pub output: String,
    /// The node the output voltage is taken over, if not the reference node.
    pub reference: Option<String>,
    /// What the output is: a voltage, or a current.
    pub quantity: Quantity,
    /// The operating point the circuit was linearized at.
    pub operating_point: OperatingPoint,
    /// The frequencies, in hertz.
    pub frequencies: Vec<f64>,
    /// The output's one-sided noise density at each frequency: in V²/Hz for a voltage, or
    /// A²/Hz for a current.
    pub densities: Vec<f64>,
    /// What each device's noise sources add to the output's density, in the order of the
    /// devices.
    pub contributions: Vec<NoiseContribution>,
    /// The source the noise is referred back to, if any.
    pub input: Option<NoiseInput>,
}

/// The independent source a noise analysis refers its output's noise back to.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseInput {
    /// The path of the source.
    pub source: String,
    /// What the source's level is: a voltage, or a current.
    pub quantity: Quantity,
    /// The gain from the source's level to the output at each frequency.
    pub gains: Vec<Complex64>,
}

impl NoiseSpectrum {
    /// The noise referred to the input at each frequency, the output's density over the
    /// gain's square: in V²/Hz for a voltage source, or A²/Hz for a current source, given
    /// an input; infinite where the gain is zero.
    #[must_use]
    pub fn referred(&self) -> Option<Vec<f64>> {
        let input = self.input.as_ref()?;
        Some(crate::measure::referred(&self.densities, &input.gains))
    }
}

/// What one noise source of a device adds to an output's noise density.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseContribution {
    /// The path of the device, such as `X1/Q1`.
    pub device: String,
    /// What makes the noise, such as `thermal` or `collector shot`.
    pub name: &'static str,
    /// What it adds to the output's density at each frequency.
    pub densities: Vec<f64>,
}

impl Simulator {
    /// Linearizes the circuit at its operating point and finds the noise density at the
    /// output at each frequency, with each device noise source's share, and the gain from
    /// the source, if any, that refers it back there.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownOutput`] when the output or its reference is no
    /// unknown, [`SimulationError::UnknownSource`] or [`SimulationError::NotASource`] when
    /// the source is not an independent source, [`SimulationError::InvalidAnalysis`] for
    /// an output over itself, a current over a node, or no frequency or one not positive,
    /// [`SimulationError::Singular`] when the linearized circuit is singular at a
    /// frequency, and the errors of [`Simulator::operating_point`].
    pub fn noise(
        &mut self,
        noise: &Noise,
        options: &Options,
    ) -> Result<NoiseSpectrum, SimulationError> {
        self.solvable()?;
        let (output, reference, quantity) =
            self.measured(&noise.output, noise.reference.as_deref())?;
        let unknowns = self.system.unknowns.len();
        let (mut input, level) = match &noise.source {
            Some(path) => {
                let index = self.source_index(path)?;
                let mut level = vec![0.0; unknowns];
                if let Some(source) = self.system.devices[index].device.as_source() {
                    source.level_derivative(&mut level);
                }
                let input = NoiseInput {
                    source: path.clone(),
                    quantity: self.source_quantity(index),
                    gains: Vec::with_capacity(noise.frequencies.len()),
                };
                (Some(input), level)
            }
            None => (None, Vec::new()),
        };
        if let Some(frequency) = noise
            .frequencies
            .iter()
            .find(|frequency| !(**frequency > 0.0 && frequency.is_finite()))
        {
            return Err(SimulationError::InvalidAnalysis(format!(
                "noise frequencies must be positive, not {frequency:?}"
            )));
        }
        let operating_point = self.linearize(&noise.frequencies, options)?;
        let state = Input::new(&operating_point.values, None, options.gmin);
        let mut sources: Vec<NoiseSource> = Vec::new();
        let mut contributions = Vec::new();
        for placed in &self.system.devices {
            let before = sources.len();
            placed.device.noise(&state, &mut sources);
            contributions.extend(sources[before..].iter().map(|source| NoiseContribution {
                device: placed.location().path.clone(),
                name: source.name,
                densities: Vec::with_capacity(noise.frequencies.len()),
            }));
        }
        let mut linearization = Linearization::new(&self.solver);
        let mut densities = Vec::with_capacity(noise.frequencies.len());
        for &frequency in &noise.frequencies {
            let mut weights = vec![Complex64::default(); unknowns];
            for (node, sign) in [(output, 1.0), (reference, -1.0)] {
                if let Some(unknown) = node {
                    weights[unknown.index()] += sign;
                }
            }
            linearization
                .factor(&self.workspace.contributions, frequency)?
                .solve_transpose(&mut weights)?;
            let weight = |node: Node| node.map_or(Complex64::default(), |u| weights[u.index()]);
            let mut total = 0.0;
            for (source, contribution) in sources.iter().zip(&mut contributions) {
                let [positive, negative] = source.equations;
                let density =
                    (weight(positive) - weight(negative)).norm_sqr() * source.density(frequency);
                contribution.densities.push(density);
                total += density;
            }
            densities.push(total);
            if let Some(input) = &mut input {
                let gain = weights.iter().zip(&level).map(|(z, &d)| z * d);
                input.gains.push(-gain.sum::<Complex64>());
            }
        }
        Ok(NoiseSpectrum {
            output: noise.output.clone(),
            reference: noise.reference.clone(),
            quantity,
            operating_point,
            frequencies: noise.frequencies.clone(),
            densities,
            contributions,
            input,
        })
    }
}
