//! Adjoint small-signal output noise analysis across all device noise sources.
//!
//! At each frequency, the linearized circuit satisfies `(G + jωC) x = -b` for an
//! excitation `b`. Solving the transposed system `(G + jωC)^T z = e` (with `e` selecting
//! the output minus its optional reference) yields adjoint weights such that excitation `b`
//! perturbs the output by `-z^T b`. Each independent noise source between equations `+` and
//! `-` contributes `|z_+ - z_-|^2` times its spectral density, and dividing total output
//! noise by squared source gain gives input-referred noise.

use jolty_models::{Input, Nature, Node, NoiseSource};
use jolty_numerics::Complex64;
use serde::Deserialize;

use crate::ac::{over_frequencies, selector_vector};
use crate::{OperatingPointResult, Options, SimulationError, Simulator};

/// A noise analysis: the noise density at an output, at each frequency, and how much each
/// device's noise sources add to it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Noise {
    /// The output, an unknown by name, such as `out` or `V1:current`, or ground's name.
    pub output: String,
    /// The node the output voltage is taken over, for a differential output; by default
    /// ground.
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
pub struct NoiseResult {
    /// The output's unknown, by name.
    pub output: String,
    /// The node the output voltage is taken over, if not ground.
    pub reference: Option<String>,
    /// Physical quantity of the output (voltage or current).
    pub nature: Nature,
    /// Operating point around which the circuit was linearized.
    pub operating_point: OperatingPointResult,
    /// Analysis frequencies in hertz.
    pub frequencies: Vec<f64>,
    /// One-sided output noise spectral density at each frequency (V²/Hz or A²/Hz).
    pub densities: Vec<f64>,
    /// Per-source contributions to output noise density, in device placement order.
    pub contributions: Vec<NoiseContribution>,
    /// Optional input source used for input-referred noise.
    pub input: Option<NoiseInput>,
}

/// Independent source used to compute input-referred noise.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseInput {
    /// Instance path of the reference source.
    pub source: String,
    /// Physical quantity of the source excitation (voltage or current).
    pub nature: Nature,
    /// Complex transfer gain from the source to the output at each frequency.
    pub gains: Vec<Complex64>,
}

impl NoiseResult {
    /// Computes input-referred noise spectral density at each frequency (`densities / |gain|²`),
    /// returning `None` when no input source was specified (infinite where gain is zero).
    #[must_use]
    pub fn referred(&self) -> Option<Vec<f64>> {
        let input = self.input.as_ref()?;
        Some(crate::measure::referred(&self.densities, &input.gains))
    }
}

/// Spectral noise contribution of a single device noise source at the output.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseContribution {
    /// Instance path of the device, such as `X1/Q1`.
    pub instance: String,
    /// Noise mechanism name, such as `thermal` or `collector shot`.
    pub name: &'static str,
    /// Output noise spectral density added by this source at each frequency.
    pub densities: Vec<f64>,
}

impl Simulator {
    /// Collects all device noise sources at `state` in placement order along with
    /// preallocated [`NoiseContribution`] buffers for `frequencies` points.
    fn noise_sources(
        &self,
        state: &Input<'_>,
        frequencies: usize,
    ) -> (Vec<NoiseSource>, Vec<NoiseContribution>) {
        let mut sources: Vec<NoiseSource> = Vec::new();
        let mut contributions = Vec::new();
        for (index, placed) in self.elaborated.devices.iter().enumerate() {
            let before = sources.len();
            let groups = &self.elaborated.groups;
            self.state.runs.noise(groups, index, state, &mut sources);
            contributions.extend(sources[before..].iter().map(|source| NoiseContribution {
                instance: placed.location().path.clone(),
                name: source.name,
                densities: Vec::with_capacity(frequencies),
            }));
        }
        (sources, contributions)
    }

    /// Computes output noise spectral density, per-source contributions, and optional
    /// input-referred gain across all frequencies in `noise`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::UnknownOutput`] for missing outputs or references,
    /// [`SimulationError::UnknownSource`] or [`SimulationError::NotASource`] for invalid
    /// input sources, [`SimulationError::InvalidAnalysis`] for invalid differential outputs
    /// or non-positive frequencies, [`SimulationError::Singular`] for singular linearizations,
    /// and any error from [`Simulator::operating_point`].
    pub fn noise(
        &mut self,
        noise: &Noise,
        options: &Options,
    ) -> Result<NoiseResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let (output, reference, nature) =
                self.measured(&noise.output, noise.reference.as_deref())?;
            let unknowns = self.elaborated.unknowns.len();
            let (mut input, level) = match &noise.source {
                Some(path) => {
                    let index = self.source_index(path)?;
                    let level = self.source_excitation(index);
                    let input = NoiseInput {
                        source: path.clone(),
                        nature: self.source_nature(index),
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
                    "noise frequencies require positive finite values; got {frequency:?}"
                )));
            }
            let operating_point = self.linearize(&noise.frequencies, options)?;
            let state = Input::new(&operating_point.values, None, options.gmin);
            let (sources, mut contributions) = self.noise_sources(&state, noise.frequencies.len());
            let referred = input.is_some();
            let found = over_frequencies(
                self.linearization(),
                &self.state.workspace.contributions,
                &noise.frequencies,
                &options.parallelism,
                &|factors, _, frequency| {
                    let mut weights: Vec<Complex64> = selector_vector(unknowns, output, reference);
                    factors.solve_transpose(&mut weights)?;
                    let weight =
                        |node: Node| node.map_or(Complex64::default(), |u| weights[u.index()]);
                    let mut total = 0.0;
                    let mut shares = Vec::with_capacity(sources.len());
                    for source in &sources {
                        let [positive, negative] = source.equations;
                        let density = (weight(positive) - weight(negative)).norm_sqr()
                            * source.density(frequency);
                        shares.push(density);
                        total += density;
                    }
                    let gain = referred.then(|| {
                        let gain = weights.iter().zip(&level).map(|(z, &d)| z * d);
                        -gain.sum::<Complex64>()
                    });
                    Ok((total, shares, gain))
                },
            )?;
            let mut densities = Vec::with_capacity(noise.frequencies.len());
            for (total, shares, gain) in found {
                densities.push(total);
                for (contribution, density) in contributions.iter_mut().zip(shares) {
                    contribution.densities.push(density);
                }
                if let (Some(input), Some(gain)) = (&mut input, gain) {
                    input.gains.push(gain);
                }
            }
            Ok(NoiseResult {
                output: noise.output.clone(),
                reference: noise.reference.clone(),
                nature,
                operating_point,
                frequencies: noise.frequencies.clone(),
                densities,
                contributions,
                input,
            })
        })
    }
}
