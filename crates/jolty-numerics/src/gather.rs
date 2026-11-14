//! Assembly in a fixed order: each value of a system, a residual or a stored entry of its
//! matrix, gathered as the sum of the contributions a slab holds for it, in the order a
//! [`GatherMap`] lists them.
//!
//! Whoever writes the slab writes each contribution to a place of its own, so writers never
//! share a value, and the gather adds a target's contributions one after another in the
//! map's order. Splitting the targets among threads changes nothing of that order, so the
//! sums are the same bits however the work is split.

/// The position that stands for no target, or no source: the reference node's row, or an
/// entry with the reference in its row or its column.
pub const UNMAPPED: u32 = u32::MAX;

/// How a gather folds a target's sources into it: adding each, or raising the target to the
/// largest of them, as [`f64::max`] takes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fold {
    /// Adds each source to the target.
    Add,
    /// Raises the target to each source larger than it.
    Raise,
}

/// For each target, the positions in a slab of the values that add into it, in the order
/// they add: a compressed row of sources per target.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GatherMap {
    /// Where each target's sources begin, and after the last target's, where they end.
    starts: Vec<usize>,
    /// The position in the slab of each source, target after target.
    sources: Vec<u32>,
}

impl GatherMap {
    /// The map onto `targets` targets of the sources `pairs` lists as `(target, source)`,
    /// each target's sources in the order `pairs` gives them. A pair whose target is
    /// [`UNMAPPED`] or not below `targets` adds nowhere and is left out.
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

    /// The number of targets.
    #[must_use]
    pub fn targets(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// The number of sources, over every target.
    #[must_use]
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// Whether no target has a source.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// The positions in the slab of the sources of `target`, in the order they add.
    #[must_use]
    pub fn sources_of(&self, target: usize) -> &[u32] {
        match (self.starts.get(target), self.starts.get(target + 1)) {
            (Some(&start), Some(&end)) => &self.sources[start..end],
            _ => &[],
        }
    }

    /// Adds to each target of `into` the values its sources hold in `slab`, one after
    /// another in the map's order.
    ///
    /// # Panics
    ///
    /// Panics if `into` has fewer values than the map has targets, or a source lies
    /// outside `slab`.
    pub fn add(&self, slab: &[f64], into: &mut [f64]) {
        self.add_from(0, slab, into);
    }

    /// Raises each target of `into` to the largest of the values its sources hold in
    /// `slab`, as [`f64::max`] takes them.
    ///
    /// # Panics
    ///
    /// Panics as [`GatherMap::add`] does.
    pub fn raise(&self, slab: &[f64], into: &mut [f64]) {
        self.raise_from(0, slab, into);
    }

    /// Folds the values their sources hold in each of the `N` slabs `slabs` into the
    /// targets of the plane of `into` of the same place, as its fold of `folds` says: added
    /// as [`GatherMap::add`] adds them, or raised to as [`GatherMap::raise`] raises them,
    /// one after another in the map's order. The targets split among the threads of the
    /// current rayon pool in chunks of `per_task`, and each target's sources are read once
    /// for every plane: the same bits as each plane folded alone, for any number of threads
    /// and any chunk.
    ///
    /// # Panics
    ///
    /// Panics as [`GatherMap::add`] does, for any plane.
    pub fn fold_in_parallel<const N: usize>(
        &self,
        slabs: [&[f64]; N],
        into: [&mut [f64]; N],
        folds: [Fold; N],
        per_task: usize,
    ) {
        let (targets, per_task) = (self.targets(), per_task.max(1));
        let mut rest = into.map(|plane| &mut plane[..targets]);
        rayon::scope(|scope| {
            let mut first = 0;
            while first < targets {
                let count = per_task.min(targets - first);
                let chunk = rest.each_mut().map(|plane| {
                    let (head, tail) = std::mem::take(plane).split_at_mut(count);
                    *plane = tail;
                    head
                });
                scope.spawn(move |_| self.fold_from(first, slabs, chunk, folds));
                first += count;
            }
        });
    }

    /// Folds their sources into the targets from `first` on, held by `into`, each plane as
    /// its fold says.
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
            let mut values: [f64; N] = std::array::from_fn(|plane| into[plane][target]);
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

    /// Adds their sources to the targets from `first` on, held by `into`.
    fn add_from(&self, first: usize, slab: &[f64], into: &mut [f64]) {
        let starts = &self.starts[first..];
        for (value, bounds) in into.iter_mut().zip(starts.windows(2)) {
            for &source in &self.sources[bounds[0]..bounds[1]] {
                *value += slab[source as usize];
            }
        }
    }

    /// Raises the targets from `first` on, held by `into`, to their sources.
    fn raise_from(&self, first: usize, slab: &[f64], into: &mut [f64]) {
        let starts = &self.starts[first..];
        for (value, bounds) in into.iter_mut().zip(starts.windows(2)) {
            for &source in &self.sources[bounds[0]..bounds[1]] {
                *value = value.max(slab[source as usize]);
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
    }

    proptest! {
        /// A sum is the sources added one after another in the map's order, the same bits
        /// whatever chunks the targets are split into and however many threads take them.
        #[test]
        fn any_split_gives_the_same_bits(
            pairs in prop::collection::vec((0..40u32, any::<f64>()), 0..400),
            per_task in 1..50usize,
            threads in 1..6usize,
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
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let mut largest = start.clone();
            map.raise(&slab, &mut largest);
            // The sums and the largest folded at once, and a second plane of sums, of the
            // sources halved.
            let halved: Vec<f64> = slab.iter().map(|value| value / 2.0).collect();
            let mut halves = start.clone();
            map.add(&halved, &mut halves);
            let (mut parallel, mut raised, mut both) = (start.clone(), start.clone(), start);
            pool.install(|| {
                map.fold_in_parallel(
                    [&slab, &slab, &halved],
                    [&mut parallel, &mut raised, &mut both],
                    [Fold::Add, Fold::Raise, Fold::Add],
                    per_task,
                );
            });
            let bits = |values: &[f64]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
            prop_assert_eq!(bits(&serial), bits(&expected));
            prop_assert_eq!(bits(&parallel), bits(&expected));
            prop_assert_eq!(bits(&raised), bits(&largest));
            prop_assert_eq!(bits(&both), bits(&halves));
        }
    }
}
