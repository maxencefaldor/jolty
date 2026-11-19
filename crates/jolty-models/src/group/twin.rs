//! Forward-mode parameter and unknown differentiation of individual device instances.

use std::collections::BTreeMap;
use std::fmt;

use jolty_numerics::UNMAPPED;

use super::kind::Kind;
use crate::{Gradient, Input, Local, Stamp};

/// Forward-mode differentiation directions (seeds) for a [`Twin`], targeting either
/// parameter paths (as named by [`jolty_ir::numbers`]) or circuit unknowns.
///
/// A seed moving a single quantity at rate `1.0` computes a partial derivative; a seed
/// moving multiple quantities computes a directional tangent derivative.
#[derive(Debug, Clone, PartialEq)]
pub struct Seeds {
    /// Parameter path paired with directional rates across all seeds.
    numbers: Vec<(String, Vec<f64>)>,
    count: usize,
    /// Circuit unknowns seeded at unit rate in seed order when differentiating unknowns.
    seeded: Vec<u32>,
}

impl Seeds {
    /// Maximum number of seeds per twin; twins are compiled for widths 1, 2, 4, and 8 and
    /// round intermediate counts up to the next compiled width.
    pub const MAX: usize = 8;

    /// Builds unit parameter seeds moving each path in `paths` independently along a
    /// dedicated direction (repeated paths accumulate rates across seeds).
    ///
    /// A twin accepts between `1` and [`Seeds::MAX`] seeds; empty or oversized sets return
    /// `None` from [`Runs::twin`](crate::Runs::twin).
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

    /// Builds unit unknown seeds for up to [`Seeds::MAX`] circuit `unknowns` with parameters
    /// held constant, yielding second-order equation derivatives with respect to unknowns.
    #[must_use]
    pub fn unknowns(unknowns: Vec<u32>) -> Option<Self> {
        let count = unknowns.len();
        (1..=Self::MAX).contains(&count).then_some(Self {
            numbers: Vec::new(),
            count,
            seeded: unknowns,
        })
    }

    /// Builds `count` tangent seeds (up to [`Seeds::MAX`]) with parameters held constant for
    /// use with [`Twin::evaluate_along`].
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

    /// Builds `count` parameter seeds (1 to [`Seeds::MAX`]) from `(path, rates)` entries,
    /// summing rates for duplicate paths in order; returns `None` on invalid `count` or rate
    /// length.
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

    /// Returns the number of differentiation directions.
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Returns parameter paths paired with rate arrays zero-padded to width `SEEDS`.
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

/// Equation contribution plane queried from an evaluated twin.
#[derive(Debug, Clone, Copy)]
pub(super) enum Contribution {
    Residual,
    Charge,
    Conductance,
    Capacitance,
}

/// Type-erased differentiable instance evaluator backed by a dedicated gradient slab.
pub(super) trait Twinned: fmt::Debug + Send + Sync {
    /// Evaluates the twin at `at` into its dedicated gradient block.
    fn evaluate(&mut self, at: &Local<'_>);

    /// Returns `(value, derivatives)` for `contribution` at local `index` from the most
    /// recent evaluation.
    fn number(&self, contribution: Contribution, index: usize) -> Option<(f64, &[f64])>;
}

/// Typed sensitivity twin for kernel `K` with compile-time derivative width `SEEDS` and
/// dedicated gradient storage for local unknowns and entries.
pub(super) struct KernelTwin<K: Kind, const SEEDS: usize> {
    shape: K::Shape,
    twin: K::Twin<SEEDS>,
    rows: [Vec<Gradient<SEEDS>>; 3],
    entries: [Vec<Gradient<SEEDS>>; 2],
    /// Active seed count within `SEEDS`; trailing lanes stay zero.
    seeds: usize,
}

impl<K: Kind, const SEEDS: usize> KernelTwin<K, SEEDS> {
    /// Constructs a boxed [`Twinned`] evaluator for `shape`, `parameters`, and `temperature`
    /// along `seeds`, sized for `unknowns` local unknowns and `stride` local entries.
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

/// Differentiable single-instance evaluator propagating forward-mode derivatives along
/// parameter or unknown seeds into a dedicated local block mapped to global circuit slots.
#[derive(Debug)]
pub struct Twin {
    evaluated: Box<dyn Twinned>,
    /// Global unknown index for each local unknown, or [`UNMAPPED`] for ground.
    nodes: Vec<u32>,
    /// Stored sparsity slot index for each local entry, or [`UNMAPPED`] for ground.
    slots: Vec<u32>,
    /// Global unknown index moved by each seed when seeding unknowns, or empty when seeding
    /// parameters.
    seeded: Vec<u32>,
    seeds: usize,
}

/// Evaluated twin contribution pairing the primal value with directional derivatives across
/// all active seeds.
pub type TwinNumber<'a> = (f64, &'a [f64]);

impl Twin {
    /// Builds a twin from a typed evaluator, global node and entry slot maps, and `seeds`.
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

    /// Returns the number of active differentiation seeds.
    #[must_use]
    pub const fn seeds(&self) -> usize {
        self.seeds
    }

    /// Evaluates the twin at `input`.
    pub fn evaluate(&mut self, input: &Input<'_>) {
        let local = input.local(&self.nodes).seeding(&self.seeded);
        self.evaluated.evaluate(&local);
    }

    /// Evaluates the twin at `input` while seeding global unknowns along each tangent vector
    /// in `tangents`.
    pub fn evaluate_along(&mut self, input: &Input<'_>, tangents: &[Vec<f64>]) {
        let local = input.local(&self.nodes).along(tangents);
        self.evaluated.evaluate(&local);
    }

    /// Returns the global unknown moved by each seed when seeding unknowns ([`UNMAPPED`] for
    /// ground).
    #[must_use]
    pub fn seeded(&self) -> &[u32] {
        &self.seeded
    }

    /// Iterates over non-ground `(unknown, (f, df))` residual contributions in local unknown
    /// order from the most recent evaluation.
    pub fn residual(&self) -> impl Iterator<Item = (usize, TwinNumber<'_>)> + '_ {
        self.by_unknown(Contribution::Residual)
    }

    /// Iterates over non-ground `(unknown, (q, dq))` charge contributions in local unknown
    /// order, mirroring [`Twin::residual`].
    pub fn charge(&self) -> impl Iterator<Item = (usize, TwinNumber<'_>)> + '_ {
        self.by_unknown(Contribution::Charge)
    }

    /// Iterates over non-ground stored sparsity slots stamped by the twin in local entry
    /// order.
    pub fn slots(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots
            .iter()
            .copied()
            .filter(|&slot| slot != UNMAPPED)
            .map(|slot| slot as usize)
    }

    /// Iterates over non-ground `(slot, conductance, capacitance)` entry contributions and
    /// derivatives in local entry order from the most recent evaluation.
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

    /// Iterates over non-ground `(unknown, value)` entries for `contribution` in local
    /// unknown order.
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
