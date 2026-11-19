//! Deterministic fixed-order gather assembly from per-contribution slabs via [`GatherMap`].
//!
//! Each writer records contributions into dedicated slab slots without sharing state. During
//! assembly, each target accumulates its slab entries sequentially in map order, yielding
//! bit-identical results regardless of how targets are partitioned across threads.

use rayon::iter::ParallelIterator;

/// Sentinel index representing an unmapped target or source (such as ground rows and columns).
pub const UNMAPPED: u32 = u32::MAX;

/// Combination rule used by a gather pass to fold slab contributions into each target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fold {
    /// Sum of all source contributions into the target.
    Add,
    /// Maximum of all source contributions via [`f64::max`].
    Raise,
}

/// Compressed per-target index map of slab contribution slots in fixed accumulation order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GatherMap {
    /// Starting offset in `sources` for each target, plus the total length sentinel.
    starts: Vec<usize>,
    /// Slab indices of all source contributions concatenated target by target.
    sources: Vec<u32>,
}

/// Contiguous slice of targets and corresponding output planes for parallel gather splitting.
struct Part<'a, const N: usize> {
    first: usize,
    planes: [&'a mut [f64]; N],
}

impl GatherMap {
    /// Builds a gather map for `targets` destinations from `(target, source)` pairs, keeping
    /// each target's sources in input order and ignoring [`UNMAPPED`] or out-of-range targets.
    #[must_use]
    pub fn new(targets: usize, pairs: impl Iterator<Item = (u32, u32)> + Clone) -> Self {
        let index = |target: u32| {
            usize::try_from(target)
                .ok()
                .filter(|&index| index < targets)
        };
        let mut starts = vec![0; targets + 1];
        for (target, _) in pairs.clone() {
            if let Some(target) = index(target) {
                starts[target + 1] += 1;
            }
        }
        for target in 0..targets {
            starts[target + 1] += starts[target];
        }
        let mut next = starts.clone();
        let mut sources = vec![0; starts[targets]];
        for (target, source) in pairs {
            if let Some(target) = index(target) {
                sources[next[target]] = source;
                next[target] += 1;
            }
        }
        Self { starts, sources }
    }

    /// Returns the number of target slots.
    #[must_use]
    pub fn targets(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Returns the total number of mapped source contributions across all targets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Returns whether the gather map contains no source contributions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Returns the slab indices contributing to `target` in accumulation order.
    #[must_use]
    pub fn sources_of(&self, target: usize) -> &[u32] {
        match (self.starts.get(target), self.starts.get(target + 1)) {
            (Some(&start), Some(&end)) => &self.sources[start..end],
            _ => &[],
        }
    }

    /// Adds each target's mapped slab contributions into `into` in map order.
    ///
    /// # Panics
    ///
    /// Panics if `into` has fewer entries than [`GatherMap::targets`] or if a source index
    /// lies outside `slab`.
    pub fn add(&self, slab: &[f64], into: &mut [f64]) {
        self.fold_into(slab, into, |value, term| value + term);
    }

    /// Updates each target in `into` to the maximum of its existing value and its mapped slab
    /// contributions via [`f64::max`].
    ///
    /// # Panics
    ///
    /// Panics under the same bounds conditions as [`GatherMap::add`].
    pub fn raise(&self, slab: &[f64], into: &mut [f64]) {
        self.fold_into(slab, into, f64::max);
    }

    /// Folds `N` parallel slabs into `N` destination planes from zero according to `folds`.
    ///
    /// Each target traverses its source list once across all planes in map order using
    /// [`GatherMap::add`] or [`GatherMap::raise`] semantics. Targets split adaptively across
    /// rayon threads by source count, producing bit-identical results on any thread count.
    ///
    /// # Panics
    ///
    /// Panics under the same bounds conditions as [`GatherMap::add`] for any plane.
    pub fn fold_in_parallel<const N: usize>(
        &self,
        slabs: [&[f64]; N],
        into: [&mut [f64]; N],
        folds: [Fold; N],
    ) {
        let targets = self.targets();
        let whole = Part {
            first: 0,
            planes: into.map(|plane| &mut plane[..targets]),
        };
        rayon::iter::split(whole, |part| self.halve(part))
            .for_each(|part| self.fold_from(part.first, slabs, part.planes, folds));
    }

    /// Splits `part` at the target balancing source counts, or returns `(part, None)` when
    /// `part` contains fewer than two targets.
    fn halve<'a, const N: usize>(&self, part: Part<'a, N>) -> (Part<'a, N>, Option<Part<'a, N>>) {
        let count = part.planes.first().map_or(0, |plane| plane.len());
        if count < 2 {
            return (part, None);
        }
        let starts = &self.starts[part.first..=part.first + count];
        let middle = starts[0] + (starts[count] - starts[0]) / 2;
        // Pick the first interior target whose sources start at or past the midpoint.
        let at = starts
            .partition_point(|&start| start < middle)
            .clamp(1, count - 1);
        let mut planes = part.planes;
        let later = planes.each_mut().map(|plane| {
            let (head, tail) = std::mem::take(plane).split_at_mut(at);
            *plane = head;
            tail
        });
        let first = part.first;
        (
            Part { first, planes },
            Some(Part {
                first: first + at,
                planes: later,
            }),
        )
    }

    /// Folds targets starting at `first` from zero into `into` according to `folds`.
    fn fold_from<const N: usize>(
        &self,
        first: usize,
        slabs: [&[f64]; N],
        mut into: [&mut [f64]; N],
        folds: [Fold; N],
    ) {
        let starts = &self.starts[first..];
        let count = into.first().map_or(0, |plane| plane.len());
        for (target, bounds) in starts.windows(2).take(count).enumerate() {
            let mut values = [0.0; N];
            for &source in &self.sources[bounds[0]..bounds[1]] {
                for ((value, slab), fold) in values.iter_mut().zip(slabs).zip(folds) {
                    let term = slab[source as usize];
                    *value = match fold {
                        Fold::Add => *value + term,
                        Fold::Raise => value.max(term),
                    };
                }
            }
            for (plane, value) in into.iter_mut().zip(values) {
                plane[target] = value;
            }
        }
    }

    /// Folds each target's sources from `slab` into `into` using `combine`.
    #[inline]
    fn fold_into(&self, slab: &[f64], into: &mut [f64], combine: impl Fn(f64, f64) -> f64) {
        let targets = self.targets();
        let len = into.len();
        assert!(
            len >= targets,
            "destination has {len} entries for {targets} targets"
        );
        for (value, bounds) in into[..targets].iter_mut().zip(self.starts.windows(2)) {
            for &source in &self.sources[bounds[0]..bounds[1]] {
                *value = combine(*value, slab[source as usize]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    #[test]
    fn lists_each_targets_sources_in_the_order_given() {
        let pairs = [(2, 0), (0, 1), (UNMAPPED, 2), (2, 3), (0, 4), (5, 5)];
        let map = GatherMap::new(3, pairs.into_iter());
        assert_eq!(map.targets(), 3);
        assert_eq!(map.len(), 4);
        assert_eq!(map.sources_of(0), [1, 4]);
        assert_eq!(map.sources_of(1), [0u32; 0]);
        assert_eq!(map.sources_of(2), [0, 3]);
        let slab = [1.0, 10.0, 100.0, 1000.0, 1e4, 1e5];
        let mut sums = [0.5; 3];
        map.add(&slab, &mut sums);
        assert_eq!(sums, [0.5 + 10.0 + 1e4, 0.5, 0.5 + 1.0 + 1000.0]);
        let mut largest = [0.0, 7.0, 2.0];
        map.raise(&slab, &mut largest);
        assert_eq!(largest, [1e4, 7.0, 1000.0]);
        // Extra trailing entries in `into` stay untouched.
        let mut padded = [0.0, 0.0, 0.0, 42.0];
        map.add(&slab, &mut padded);
        assert_eq!(padded, [10.0 + 1e4, 0.0, 1.0 + 1000.0, 42.0]);
    }

    #[test]
    #[should_panic(expected = "destination has 2 entries for 3 targets")]
    fn add_panics_when_destination_is_shorter_than_targets() {
        let map = GatherMap::new(3, [(0, 0)].into_iter());
        let mut short = [0.0; 2];
        map.add(&[1.0], &mut short);
    }

    #[test]
    #[should_panic(expected = "destination has 1 entries for 3 targets")]
    fn raise_panics_when_destination_is_shorter_than_targets() {
        let map = GatherMap::new(3, [(0, 0)].into_iter());
        let mut short = [0.0; 1];
        map.raise(&[1.0], &mut short);
    }

    proptest! {
        /// Verifies that parallel gather produces bit-identical results to sequential gather
        /// across 1, 2, 3, and 8 threads.
        #[test]
        fn any_split_gives_the_same_bits(
            pairs in prop::collection::vec((0..40u32, any::<f64>()), 0..400),
        ) {
            let slab: Vec<f64> = pairs
                .iter()
                .map(|&(_, value)| if value.is_finite() { value % 1e6 } else { 1.0 })
                .collect();
            let sources = (0u32..).zip(&pairs).map(|(source, &(target, _))| (target, source));
            let map = GatherMap::new(40, sources);
            let start: Vec<f64> = (0..40).map(|target| f64::from(target) * 0.1).collect();
            let mut expected = start.clone();
            for (&(target, _), value) in pairs.iter().zip(&slab) {
                expected[target as usize] += value;
            }
            let mut serial = start.clone();
            map.add(&slab, &mut serial);
            prop_assert_eq!(bits(&serial), bits(&expected));
            // The sums and the largest from zero, and a second plane of sums, of the sources
            // halved, each folded alone.
            let halved: Vec<f64> = slab.iter().map(|value| value / 2.0).collect();
            let (mut sums, mut largest, mut halves) = (vec![0.0; 40], vec![0.0; 40], vec![0.0; 40]);
            map.add(&slab, &mut sums);
            map.raise(&slab, &mut largest);
            map.add(&halved, &mut halves);
            for threads in [1, 2, 3, 8] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
                // Folded at once, over whatever the targets held.
                let (mut parallel, mut raised, mut both) =
                    (start.clone(), start.clone(), start.clone());
                pool.install(|| {
                    map.fold_in_parallel(
                        [&slab, &slab, &halved],
                        [&mut parallel, &mut raised, &mut both],
                        [Fold::Add, Fold::Raise, Fold::Add],
                    );
                });
                prop_assert_eq!(bits(&parallel), bits(&sums), "{} threads", threads);
                prop_assert_eq!(bits(&raised), bits(&largest), "{} threads", threads);
                prop_assert_eq!(bits(&both), bits(&halves), "{} threads", threads);
            }
        }
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }
}
