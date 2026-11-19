//! Forward-mode parameter and unknown differentiation of individual device instances.

use std::collections::BTreeMap;
use std::fmt;

use jolty_numerics::UNMAPPED;

use super::kind::Kind;
use crate::{Gradient, Input, Local, Stamp};

/// The directions a [`Twin`] differentiates along, its seeds: numbers of an instance's
/// parameters, each at its path among them, as [`jolty_ir::numbers`] names it, with its
/// rate along each seed, or the circuit's unknowns. A seed that moves one number at rate 1
/// differentiates against that number alone; one that moves several, along that tangent of
/// them.
#[derive(Debug, Clone, PartialEq)]
pub struct Seeds {
    /// Each number's path, and its rate along each seed.
    numbers: Vec<(String, Vec<f64>)>,
    count: usize,
    /// The circuit's unknowns the seeds move at rate one, in their order, when seeding unit
    /// unknowns.
    seeded: Vec<u32>,
}

impl Seeds {
    /// The maximum number of seeds one twin carries: twins are compiled for 1, 2, 4 and this many,
    /// a twin of another count taking the next of them up.
    pub const MAX: usize = 8;

    /// A seed for each of the numbers at `paths`, which moves that number alone; a path
    /// given more than once moves along each of its seeds.
    ///
    /// A twin accepts between `1` and [`Seeds::MAX`] seeds; empty or larger sets yield
    /// `None` when passed to [`Runs::twin`](crate::Runs::twin).
    #[must_use]
    pub fn units(paths: &[&str]) -> Self {
        let count = paths.len();
        let numbers = paths
            .iter()
            .enumerate()
            .map(|(seed, &path)| {
                let mut rates = vec![0.0; count];
                if let Some(rate) = rates.get_mut(seed) {
                    *rate = 1.0;
                }
                (path.to_owned(), rates)
            })
            .collect();
        Self::new(numbers, count).unwrap_or(Self {
            numbers: Vec::new(),
            count,
            seeded: Vec::new(),
        })
    }

    /// A seed for each of the circuit's `unknowns`, at most [`Seeds::MAX`], its parameters
    /// constant: the derivatives of what an instance adds with respect to those unknowns,
    /// the second derivatives of its equations among them.
    #[must_use]
    pub fn unknowns(unknowns: Vec<u32>) -> Option<Self> {
        let count = unknowns.len();
        (1..=Self::MAX).contains(&count).then_some(Self {
            numbers: Vec::new(),
            count,
            seeded: unknowns,
        })
    }

    /// `count` seeds, at most [`Seeds::MAX`], its parameters constant, for
    /// [`Twin::evaluate_along`] to move the circuit's unknowns along `count` tangent
    /// vectors.
    #[must_use]
    pub const fn tangents(count: usize) -> Option<Self> {
        if count >= 1 && count <= Self::MAX {
            Some(Self {
                numbers: Vec::new(),
                count,
                seeded: Vec::new(),
            })
        } else {
            None
        }
    }

    /// `count` seeds, between `1` and [`Seeds::MAX`], along which each number of
    /// `numbers`, at its path, moves at its rates, one per seed, a path given more than once
    /// moving at the sum of its rates, added in the order given; `None` when `count` is out
    /// of range or a number has another count of rates.
    #[must_use]
    pub fn new(numbers: Vec<(String, Vec<f64>)>, count: usize) -> Option<Self> {
        if !(1..=Self::MAX).contains(&count)
            || numbers.iter().any(|(_, rates)| rates.len() != count)
        {
            return None;
        }
        let mut place: BTreeMap<String, usize> = BTreeMap::new();
        let mut summed: Vec<(String, Vec<f64>)> = Vec::with_capacity(numbers.len());
        for (path, rates) in numbers {
            let at = *place.entry(path.clone()).or_insert_with(|| {
                summed.push((path, vec![0.0; count]));
                summed.len() - 1
            });
            for (total, rate) in summed[at].1.iter_mut().zip(rates) {
                *total += rate;
            }
        }
        Some(Self {
            numbers: summed,
            count,
            seeded: Vec::new(),
        })
    }

    /// How many seeds there are.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Each number's path and its rates, padded with zeros to `SEEDS` seeds.
    fn padded<const SEEDS: usize>(&self) -> Vec<(&str, [f64; SEEDS])> {
        self.numbers
            .iter()
            .map(|(path, rates)| {
                let padded = std::array::from_fn(|seed| rates.get(seed).copied().unwrap_or(0.0));
                (path.as_str(), padded)
            })
            .collect()
    }
}

/// What an instance's twin adds to the circuit's equations, by kind.
#[derive(Debug, Clone, Copy)]
pub(super) enum Contribution {
    Residual,
    Charge,
    Conductance,
    Capacitance,
}

/// An instance's twin, typed, apart from its group.
pub(super) trait Twinned: fmt::Debug + Send + Sync {
    /// Evaluates the twin at `at` into its own block.
    fn evaluate(&mut self, at: &Local<'_>);

    /// The `contribution` it adds at the local unknown or local entry `index`, as last
    /// evaluated: its value, and its derivative along each seed.
    fn number(&self, contribution: Contribution, index: usize) -> Option<(f64, &[f64])>;
}

/// The twin of an instance of the kind `K`, its derivatives along `SEEDS` directions, its
/// shape, and its block: the residuals, scales and charges of its local unknowns and the
/// conductances and capacitances of its local entries, as last evaluated.
pub(super) struct KernelTwin<K: Kind, const SEEDS: usize> {
    shape: K::Shape,
    twin: K::Twin<SEEDS>,
    rows: [Vec<Gradient<SEEDS>>; 3],
    entries: [Vec<Gradient<SEEDS>>; 2],
    /// How many of the `SEEDS` directions the twin was seeded along, the rest left zero.
    seeds: usize,
}

impl<K: Kind, const SEEDS: usize> KernelTwin<K, SEEDS> {
    /// The twin of an instance of `shape` with `parameters` at `temperature` along `seeds`,
    /// with room for `sizes`, its local unknowns and its local entries.
    pub(super) fn made(
        (shape, parameters, temperature): (&K::Shape, &K::Parameters, f64),
        seeds: &Seeds,
        (unknowns, stride): (usize, usize),
    ) -> Option<Box<dyn Twinned>> {
        let twin = K::twin(shape, parameters, &seeds.padded::<SEEDS>(), temperature)?;
        let zero = |size| vec![Gradient::default(); size];
        Some(Box::new(Self {
            shape: shape.clone(),
            twin,
            rows: [zero(unknowns), zero(unknowns), zero(unknowns)],
            entries: [zero(stride), zero(stride)],
            seeds: seeds.count(),
        }))
    }
}

impl<K: Kind, const SEEDS: usize> fmt::Debug for KernelTwin<K, SEEDS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(K::NAME)
            .field("shape", &self.shape)
            .field("twin", &self.twin)
            .finish_non_exhaustive()
    }
}

impl<K: Kind, const SEEDS: usize> Twinned for KernelTwin<K, SEEDS> {
    fn evaluate(&mut self, at: &Local<'_>) {
        let [residual, scale, charge] = &mut self.rows;
        let [conductance, capacitance] = &mut self.entries;
        let mut stamp = Stamp::zeroed([residual, scale, charge], [conductance, capacitance]);
        K::evaluate_twin(&self.shape, &mut self.twin, at, &mut stamp);
    }

    fn number(&self, contribution: Contribution, index: usize) -> Option<(f64, &[f64])> {
        let [residual, _, charge] = &self.rows;
        let [conductance, capacitance] = &self.entries;
        let values = match contribution {
            Contribution::Residual => residual,
            Contribution::Charge => charge,
            Contribution::Conductance => conductance,
            Contribution::Capacitance => capacitance,
        };
        let number = values.get(index)?;
        Some((number.value, number.derivatives.get(..self.seeds)?))
    }
}

/// A device's instance computing with the derivatives of every number along some
/// directions of its parameters, its seeds: how the contributions it adds to the circuit's
/// equations move along each, which its own block holds, local to the instance, and the
/// map of its group places among the circuit's.
#[derive(Debug)]
pub struct Twin {
    evaluated: Box<dyn Twinned>,
    /// The circuit's unknown of each local unknown, or [`UNMAPPED`] for one on ground.
    nodes: Vec<u32>,
    /// The stored position of each local entry, or [`UNMAPPED`] for none.
    slots: Vec<u32>,
    /// The circuit's unknown each seed moves, in their order, or [`UNMAPPED`] for a local
    /// unknown on ground; none when the seeds are directions of the parameters.
    seeded: Vec<u32>,
    seeds: usize,
}

/// A contribution of a twin as last evaluated: its value, and its derivative along each of
/// the twin's seeds.
pub type TwinNumber<'a> = (f64, &'a [f64]);

impl Twin {
    /// Builds a twin from its typed evaluator, its local unknown and entry mappings, and
    /// its seeds.
    pub(super) fn new(
        evaluated: Box<dyn Twinned>,
        nodes: Vec<u32>,
        slots: Vec<u32>,
        seeds: &Seeds,
    ) -> Self {
        Self {
            evaluated,
            nodes,
            slots,
            seeded: seeds.seeded.clone(),
            seeds: seeds.count(),
        }
    }

    /// How many seeds the twin differentiates along.
    #[must_use]
    pub const fn seeds(&self) -> usize {
        self.seeds
    }

    /// Evaluates the twin at `input`.
    pub fn evaluate(&mut self, input: &Input<'_>) {
        let local = input.local(&self.nodes).seeding(&self.seeded);
        self.evaluated.evaluate(&local);
    }

    /// Evaluates the twin at `input`, moving the circuit's unknowns along each of
    /// `tangents`, one seed per tangent.
    pub fn evaluate_along(&mut self, input: &Input<'_>, tangents: &[Vec<f64>]) {
        let local = input.local(&self.nodes).along(tangents);
        self.evaluated.evaluate(&local);
    }

    /// The circuit's unknown each seed moves, by seed, when the seeds are local unknowns:
    /// [`UNMAPPED`] for one on ground, which moves nothing.
    #[must_use]
    pub fn seeded(&self) -> &[u32] {
        &self.seeded
    }

    /// Each residual the instance adds, by the circuit's unknown whose equation it enters,
    /// in the order of its local unknowns, those on ground left out: `f`, and
    /// `∂f/∂p · s` along each seed `s`, as last evaluated.
    pub fn residual(&self) -> impl Iterator<Item = (usize, TwinNumber<'_>)> + '_ {
        self.by_unknown(Contribution::Residual)
    }

    /// Each charge the instance adds, as [`Twin::residual`] gives the residuals: `q`, and
    /// `∂q/∂p · s` along each seed, as last evaluated.
    pub fn charge(&self) -> impl Iterator<Item = (usize, TwinNumber<'_>)> + '_ {
        self.by_unknown(Contribution::Charge)
    }

    /// Each mapped matrix slot the twin stamps into, in the order of its local entries,
    /// those on ground left out.
    pub fn slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots
            .iter()
            .copied()
            .filter(|&slot| slot != UNMAPPED)
            .map(|slot| slot as usize)
    }

    /// Each conductance and capacitance the instance adds, by the stored position it adds
    /// to, in the order of its local entries, those on ground left out: `G` and `C`,
    /// and their derivatives along each seed, as last evaluated.
    pub fn entries(&self) -> impl Iterator<Item = (usize, TwinNumber<'_>, TwinNumber<'_>)> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| **slot != UNMAPPED)
            .filter_map(|(entry, &slot)| {
                let conductance = self.evaluated.number(Contribution::Conductance, entry)?;
                let capacitance = self.evaluated.number(Contribution::Capacitance, entry)?;
                Some((slot as usize, conductance, capacitance))
            })
    }

    /// Each `contribution` at a local unknown, by the circuit's unknown of the local
    /// unknown, those on ground left out.
    fn by_unknown(
        &self,
        contribution: Contribution,
    ) -> impl Iterator<Item = (usize, TwinNumber<'_>)> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| **node != UNMAPPED)
            .filter_map(move |(local, &node)| {
                Some((node as usize, self.evaluated.number(contribution, local)?))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::Seeds;

    #[test]
    fn a_path_given_twice_moves_at_the_sum_of_its_rates() {
        let numbers = vec![
            ("w".to_owned(), vec![1.0, -0.5]),
            ("l".to_owned(), vec![0.0, 1.0]),
            ("w".to_owned(), vec![0.25, 2.0]),
        ];
        let seeds = Seeds::new(numbers, 2).expect("two seeds");
        assert_eq!(
            seeds.padded::<2>(),
            vec![("w", [1.25, 1.5]), ("l", [0.0, 1.0])]
        );
        let units = Seeds::units(&["w", "w"]);
        assert_eq!(units.padded::<2>(), vec![("w", [1.0, 1.0])]);
    }
}
