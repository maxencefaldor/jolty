//! Variants: the values of every run of a [batch](crate::batch), each a pure function of the
//! run's index.
//!
//! [`Explicit`] variants list their values; a [`Grid`] steps axes of parameters or the
//! temperature in mixed radix, the last axis fastest; a [`MonteCarlo`] analysis draws each
//! toleranced number from its seed, the sample's index and the number's name; and a
//! [`Product`] runs every variant of one at every variant of another, so corners, samples
//! and a sweep make one batch.
//!
//! A Monte Carlo draw depends on nothing but its seed, its sample's index and a stable key:
//! the FNV-1a hash of the instance's path and the parameter's, or of the lot's name and the
//! parameter's. So sample `i` drawn alone is sample `i` of any batch, however many threads
//! run it, and a part added to the circuit leaves every other part's draws as they were.

use jolty_ir::{Circuit, Distribution, Of, Tolerance};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::StandardNormal;

use crate::batch::Variants;
use crate::elaborate::{ElaborationError, ElaborationProblem};
use crate::{Elaborated, ParameterId, Values};

/// Variants that list the values of each run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Explicit(pub Vec<Values>);

impl Variants for Explicit {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }

    fn values(&self, index: u64) -> Values {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.0.get(index))
            .cloned()
            .unwrap_or_default()
    }
}

/// What an axis of a [`Grid`] steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    /// A number of a device's parameters.
    Parameter(ParameterId),
    /// The devices' temperature, in kelvins.
    Temperature,
}

/// Variants on a grid: every combination of the values of its axes, in mixed radix with
/// the last axis fastest, so that the runs of one value of every axis but the last are
/// consecutive.
///
/// A warm grid runs each line along its last axis as a chain, each run starting from the
/// solution of the run before it, as a DC sweep does from one level to the next.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    axes: Vec<(Axis, Vec<f64>)>,
    warm: bool,
    len: u64,
}

impl Grid {
    /// The grid of `axes`, each what it steps and its values in order, `warm` when each line
    /// along the last is a chain. A grid of no axes has one point, which replaces nothing.
    ///
    /// # Errors
    ///
    /// Returns [`VariantsError::Empty`] for an axis of no values, and
    /// [`VariantsError::TooMany`] when the grid has more points than a `u64` counts.
    pub fn new(axes: Vec<(Axis, Vec<f64>)>, warm: bool) -> Result<Self, VariantsError> {
        let mut len: u64 = 1;
        for (index, (_, values)) in axes.iter().enumerate() {
            if values.is_empty() {
                return Err(VariantsError::Empty(index));
            }
            len = len
                .checked_mul(values.len() as u64)
                .ok_or(VariantsError::TooMany)?;
        }
        Ok(Self { axes, warm, len })
    }
}

impl Variants for Grid {
    fn len(&self) -> u64 {
        self.len
    }

    fn values(&self, index: u64) -> Values {
        let mut values = Values::default();
        let mut rest = index;
        for (axis, steps) in self.axes.iter().rev() {
            let count = steps.len() as u64;
            // The remainder is below the axis's length, a `usize`.
            let step = usize::try_from(rest % count).unwrap_or_default();
            rest /= count;
            match *axis {
                Axis::Parameter(parameter) => values.set(parameter, steps[step]),
                Axis::Temperature => values.temperature = Some(steps[step]),
            }
        }
        values
    }

    fn chain(&self) -> u64 {
        match self.axes.last() {
            Some((_, steps)) if self.warm => steps.len() as u64,
            _ => 1,
        }
    }
}

/// Every variant of `A` at every variant of `B`: run `i` is variant `i / len(B)` of `A`,
/// then variant `i % len(B)` of `B`, whose values replace `A`'s where both give one. Its
/// chains are those of `B`.
#[derive(Debug, Clone, PartialEq)]
pub struct Product<A, B> {
    first: A,
    second: B,
    len: u64,
}

impl<A: Variants, B: Variants> Product<A, B> {
    /// Every variant of `first` at every variant of `second`.
    ///
    /// # Errors
    ///
    /// Returns [`VariantsError::TooMany`] when there are more runs than a `u64` counts.
    pub fn new(first: A, second: B) -> Result<Self, VariantsError> {
        let len = first
            .len()
            .checked_mul(second.len())
            .ok_or(VariantsError::TooMany)?;
        Ok(Self { first, second, len })
    }

    /// The variants of which it runs every one at every one: the first, then the second.
    #[must_use]
    pub const fn parts(&self) -> (&A, &B) {
        (&self.first, &self.second)
    }
}

impl<A: Variants, B: Variants> Variants for Product<A, B> {
    fn len(&self) -> u64 {
        self.len
    }

    fn values(&self, index: u64) -> Values {
        let count = self.second.len().max(1);
        let mut values = self.first.values(index / count);
        values.extend(&self.second.values(index % count));
        values
    }

    fn chain(&self) -> u64 {
        self.second.chain()
    }
}

/// Why variants cannot be made.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VariantsError {
    /// An axis of a grid has no values, by its index.
    #[error("axis {0} of the grid has no values")]
    Empty(usize),
    /// The variants are more than a `u64` counts.
    #[error("the variants are more than 2⁶⁴ − 1")]
    TooMany,
}

/// A number of a device instance that varies within its tolerance: where it is, its
/// nominal value, and its tolerance.
#[derive(Debug, Clone, PartialEq)]
pub struct Toleranced {
    /// The instance's path, such as `X1/R1`.
    pub path: String,
    /// The number's path among the instance's parameters, such as `resistance` or
    /// `waveform.dc`.
    pub name: String,
    /// The number the circuit writes.
    pub nominal: f64,
    /// How far its values spread.
    pub tolerance: Tolerance,
}

/// What a draw is drawn for: a part's own factor, or its lot's.
#[derive(Debug, Clone, Copy)]
#[repr(u64)]
enum Drawn {
    Part = 0,
    Lot = 1,
}

impl Toleranced {
    /// The number's value in the sample `index` of a Monte Carlo analysis seeded with
    /// `seed`: its nominal value times its own factor and, if it is in a lot, the lot's.
    ///
    /// Each factor is drawn from a `ChaCha20` generator seeded with four little-endian 64-bit
    /// words: the seed, the sample's index, a stable key, and whether it is a part's own
    /// factor or its lot's. A part's key is the FNV-1a hash of its instance's path and the
    /// number's; a lot's, of the lot's name and the number's, so every part of the lot
    /// draws the lot's factor alike. A factor is `1 + d` for a linear spread, `d` uniform
    /// within the tolerance or normal with the tolerance three standard deviations, and
    /// `(1 + relative)^u` for a logarithmic one, `u` uniform within 1 of 0 or normal with 1
    /// three standard deviations, each drawn as numpy draws it from a uniform number in
    /// `[0, 1)` or a standard normal one.
    #[must_use]
    pub fn drawn(&self, seed: u64, index: u64) -> f64 {
        let tolerance = &self.tolerance;
        let own = generator(seed, index, key(&self.path, &self.name), Drawn::Part);
        let mut factor = factor(tolerance.distribution, tolerance.relative, own);
        if let Some(lot) = &tolerance.lot {
            let shared = generator(seed, index, key(lot.name.as_str(), &self.name), Drawn::Lot);
            factor *= self::factor(lot.distribution, lot.relative, shared);
        }
        self.nominal * factor
    }
}

/// The stable key of `first` and `second`, an instance's path and a number's, or a lot's
/// name and a number's: the FNV-1a hash of the two joined by the byte 0xFF, which no UTF-8
/// text holds, so that no two pairs join alike.
fn key(first: &str, second: &str) -> u64 {
    fnv(first.bytes().chain([0xff]).chain(second.bytes()))
}

/// The FNV-1a hash of `bytes`, 64 bits wide (Fowler, Noll and Vo; IETF draft
/// `draft-eastlake-fnv`).
fn fnv(bytes: impl IntoIterator<Item = u8>) -> u64 {
    /// FNV's offset basis and prime for 64 bits.
    const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.into_iter().fold(BASIS, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(PRIME)
    })
}

/// The generator of the draw of `drawn` for the stable `key` in the sample `index` of the
/// analysis seeded with `seed`.
fn generator(seed: u64, index: u64, key: u64, drawn: Drawn) -> ChaCha20Rng {
    let mut words = [0u8; 32];
    for (place, word) in [seed, index, key, drawn as u64].into_iter().enumerate() {
        words[8 * place..8 * place + 8].copy_from_slice(&word.to_le_bytes());
    }
    ChaCha20Rng::from_seed(words)
}

/// A factor drawn from `generator` within a spread of `relative` as `distribution` takes it,
/// as `jolty.variation` defines it.
fn factor(distribution: Distribution, relative: f64, mut generator: ChaCha20Rng) -> f64 {
    let generator = &mut generator;
    match distribution {
        Distribution::Uniform => 1.0 + uniform(generator, -relative, relative),
        Distribution::Normal => 1.0 + normal(generator, 0.0, relative / 3.0),
        Distribution::LogUniform => (1.0 + relative).powf(uniform(generator, -1.0, 1.0)),
        Distribution::LogNormal => (1.0 + relative).powf(normal(generator, 0.0, 1.0 / 3.0)),
    }
}

/// A number drawn uniformly from `[low, high)`, as numpy's `uniform` draws it: `low` plus
/// the span times a number drawn uniformly from `[0, 1)`.
fn uniform(generator: &mut ChaCha20Rng, low: f64, high: f64) -> f64 {
    let unit: f64 = generator.random();
    low + (high - low) * unit
}

/// A number drawn normally about `mean` with the standard deviation `deviation`, as numpy's
/// `normal` draws it: the mean plus the deviation times a standard normal number.
fn normal(generator: &mut ChaCha20Rng, mean: f64, deviation: f64) -> f64 {
    let standard: f64 = generator.sample(StandardNormal);
    mean + deviation * standard
}

/// Every number of every device instance of `circuit` that has a tolerance: in the order of
/// the hierarchy, each instance's numbers in the order of their paths, so that the same
/// circuit lists the same numbers.
///
/// # Errors
///
/// Returns [`Untoleranced`] for a tolerance on a parameter the instance writes no number
/// for, which a valid circuit never holds.
pub fn toleranced(circuit: &Circuit) -> Result<Vec<Toleranced>, Untoleranced> {
    let mut found = Vec::new();
    walk(circuit, circuit.top.as_str(), "", &mut found)?;
    Ok(found)
}

/// The tolerance of a parameter an instance writes no number for.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{path} gives {name} a tolerance, not a number")]
pub struct Untoleranced {
    /// The instance's path.
    pub path: String,
    /// The parameter's path among its parameters.
    pub name: String,
}

/// Appends the toleranced numbers of the instances of `component`, whose paths start with
/// `prefix`, to `found`.
fn walk(
    circuit: &Circuit,
    component: &str,
    prefix: &str,
    found: &mut Vec<Toleranced>,
) -> Result<(), Untoleranced> {
    let Some(placed) = circuit.components.get(component) else {
        return Ok(());
    };
    for instance in &placed.instances {
        let path = format!("{prefix}{}", instance.id.as_str());
        match &instance.of {
            Of::Component(inner) => walk(circuit, inner.as_str(), &format!("{path}/"), found)?,
            Of::Device(_) => {
                for (name, tolerance) in &instance.tolerances {
                    let nominal = jolty_ir::parameter(&instance.parameters, name)
                        .and_then(serde_json::Value::as_f64)
                        .ok_or_else(|| Untoleranced {
                            path: path.clone(),
                            name: name.clone(),
                        })?;
                    found.push(Toleranced {
                        path: path.clone(),
                        name: name.clone(),
                        nominal,
                        tolerance: tolerance.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Samples of a circuit's toleranced numbers: sample `i` sets each to its value
/// [drawn](Toleranced::drawn) for `i` from the seed.
#[derive(Debug, Clone, PartialEq)]
pub struct MonteCarlo {
    seed: u64,
    samples: u64,
    varied: Vec<(ParameterId, Toleranced)>,
}

impl MonteCarlo {
    /// `samples` samples of the numbers `varied` of `elaborated`, drawn from `seed`.
    ///
    /// # Errors
    ///
    /// Returns an [`ElaborationProblem::UnknownParameter`] for each number `elaborated`
    /// does not write, and [`ElaborationProblem::UnknownOverride`] for each instance it
    /// does not have.
    pub fn new(
        elaborated: &Elaborated,
        seed: u64,
        samples: u64,
        varied: Vec<Toleranced>,
    ) -> Result<Self, ElaborationError> {
        let mut problems = Vec::new();
        let mut found = Vec::with_capacity(varied.len());
        for toleranced in varied {
            match elaborated.parameter(&toleranced.path, &toleranced.name) {
                Some(parameter) => found.push((parameter, toleranced)),
                None => problems.push(match elaborated.location(&toleranced.path) {
                    Some(at) => ElaborationProblem::UnknownParameter {
                        at,
                        parameter: toleranced.name,
                    },
                    None => ElaborationProblem::UnknownOverride(toleranced.path),
                }),
            }
        }
        if problems.is_empty() {
            Ok(Self {
                seed,
                samples,
                varied: found,
            })
        } else {
            Err(ElaborationError::new(problems))
        }
    }
}

impl Variants for MonteCarlo {
    fn len(&self) -> u64 {
        self.samples
    }

    fn values(&self, index: u64) -> Values {
        let mut values = Values::default();
        for (parameter, toleranced) in &self.varied {
            values.set(*parameter, toleranced.drawn(self.seed, index));
        }
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_fnv_1a_of_the_names_joined() {
        // The test vectors of the FNV draft for 64-bit FNV-1a.
        assert_eq!(fnv(*b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv(*b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv(*b"foobar"), 0x8594_4171_f739_67e8);
        assert_eq!(key("X1/R1", "resistance"), fnv(*b"X1/R1\xffresistance"));
        assert_ne!(key("X1/R1", "resistance"), key("X1", "R1/resistance"));
    }

    #[test]
    fn a_grid_steps_its_last_axis_fastest() {
        let (first, second) = (ParameterId::new(0), ParameterId::new(3));
        let grid = Grid::new(
            vec![
                (Axis::Parameter(first), vec![1.0, 2.0]),
                (Axis::Temperature, vec![300.0, 310.0, 320.0]),
                (Axis::Parameter(second), vec![5.0]),
            ],
            true,
        )
        .unwrap();
        assert_eq!((grid.len(), grid.chain()), (6, 1));
        let values = grid.values(4);
        assert_eq!(values.patches(), [(first, 2.0), (second, 5.0)]);
        assert_eq!(values.temperature, Some(310.0));
        assert_eq!(
            Grid::new(vec![(Axis::Temperature, vec![])], false),
            Err(VariantsError::Empty(0))
        );
    }
}
