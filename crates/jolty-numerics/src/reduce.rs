//! Deterministic chunked reductions producing bit-identical results across thread counts.
//!
//! [`sum`] accumulates terms in consecutive chunks of [`CHUNK`] and sums the chunk totals in
//! order. [`sum_in_parallel`] evaluates the same chunks on the current rayon pool and combines
//! totals in identical order. [`fold`] generalizes this scheme to arbitrary accumulators,
//! avoiding rayon's non-deterministic tree reduction.

use rayon::prelude::*;

/// Number of terms accumulated per reduction chunk before combining chunk totals.
///
/// Origin: power-of-two deterministic-reduction chunk size (`2^10 = 1024` `f64` values,
/// 8 KiB) fitting in L1 data cache while amortizing rayon scheduling overhead.
pub const CHUNK: usize = 1024;

/// Sums `count` terms produced by `term` in fixed sequential chunks of [`CHUNK`].
pub fn sum(count: usize, term: impl Fn(usize) -> f64) -> f64 {
    (0..count.div_ceil(CHUNK)).fold(0.0, |total, chunk| total + chunk_sum(chunk, count, &term))
}

/// Sums `count` terms across rayon threads with at least `least` chunks per task, matching
/// [`sum`] bit for bit.
pub fn sum_in_parallel(count: usize, term: impl Fn(usize) -> f64 + Sync, least: usize) -> f64 {
    let totals: Vec<f64> = (0..count.div_ceil(CHUNK))
        .into_par_iter()
        .with_min_len(least.max(1))
        .map(|chunk| chunk_sum(chunk, count, &term))
        .collect();
    totals.into_iter().fold(0.0, |total, chunk| total + chunk)
}

/// Sums the terms belonging to chunk index `chunk` out of `count` total terms.
fn chunk_sum(chunk: usize, count: usize, term: &impl Fn(usize) -> f64) -> f64 {
    let start = chunk * CHUNK;
    (start..count.min(start + CHUNK)).fold(0.0, |total, index| total + term(index))
}

/// Folds `items` deterministically using per-item `fold` and chunk combiner `join`.
///
/// Inputs of at most [`CHUNK`] items fold sequentially into `start` inline. Larger inputs fold
/// consecutive chunks of [`CHUNK`] items (the first into `start`, subsequent chunks into
/// `zero`) and join chunk totals in order, producing identical bits whether `spread` is false
/// or parallelized across rayon threads.
#[inline]
pub fn fold<T: Sync, A: Copy + Send + Sync>(
    items: &[T],
    (start, zero): (A, A),
    fold: impl Fn(A, &T) -> A + Sync,
    join: impl Fn(A, A) -> A,
    spread: bool,
) -> A {
    if items.len() <= CHUNK {
        return items.iter().fold(start, fold);
    }
    fold_by_chunks(items, (start, zero), fold, join, spread)
}

/// Executes the multi-chunk path of [`fold`] out of line so short folds inline cleanly.
#[inline(never)]
fn fold_by_chunks<T: Sync, A: Copy + Send + Sync>(
    items: &[T],
    (start, zero): (A, A),
    fold: impl Fn(A, &T) -> A + Sync,
    join: impl Fn(A, A) -> A,
    spread: bool,
) -> A {
    let chunk = |index: usize, chunk: &[T]| {
        let from = if index == 0 { start } else { zero };
        chunk.iter().fold(from, &fold)
    };
    if !spread {
        let mut chunks = items.chunks(CHUNK).enumerate();
        let first = chunks
            .next()
            .map_or(start, |(index, first)| chunk(index, first));
        return chunks.fold(first, |total, (index, later)| {
            join(total, chunk(index, later))
        });
    }
    let totals: Vec<A> = items
        .par_chunks(CHUNK)
        .enumerate()
        .map(|(index, items)| chunk(index, items))
        .collect();
    let mut totals = totals.into_iter();
    let first = totals.next().unwrap_or(start);
    totals.fold(first, join)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// Verifies that parallel chunked summation matches single-threaded summation bit for
        /// bit across 1 to 8 threads.
        #[test]
        fn a_sum_on_many_threads_is_the_sum_on_one_bit_for_bit(
            terms in proptest::collection::vec(-1e3..1e3f64, 0..5000),
            threads in 1..9usize,
            least in 1..4usize,
        ) {
            let serial = sum(terms.len(), |index| terms[index]);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a pool");
            let parallel = pool.install(|| sum_in_parallel(terms.len(), |index| terms[index], least));
            prop_assert_eq!(serial.to_bits(), parallel.to_bits());
        }
    }

    proptest! {
        /// Verifies that chunked [`fold`] matches sequential reduction for up to [`CHUNK`]
        /// items and produces identical bits across 1, 2, 3, and 8 threads.
        #[test]
        fn a_fold_by_chunks_is_the_same_bits_on_any_number_of_threads(
            terms in proptest::collection::vec(-1e3..1e3f64, 0..5000),
            start in -1e3..1e3f64,
        ) {
            let folded = |spread| fold(&terms, (start, 0.0), |total, term| total - term, |a, b| a + b, spread);
            let chunks: Vec<f64> = terms
                .chunks(CHUNK)
                .enumerate()
                .map(|(index, chunk)| {
                    let from = if index == 0 { start } else { 0.0 };
                    chunk.iter().fold(from, |total, term| total - term)
                })
                .collect();
            let expected = chunks.iter().skip(1).fold(chunks.first().copied().unwrap_or(start), |a, b| a + b);
            let serial = folded(false);
            prop_assert_eq!(serial.to_bits(), expected.to_bits());
            if terms.len() <= CHUNK {
                let one_by_one = terms.iter().fold(start, |total, term| total - term);
                prop_assert_eq!(serial.to_bits(), one_by_one.to_bits());
            }
            for threads in [1, 2, 3, 8] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .expect("a pool");
                prop_assert_eq!(pool.install(|| folded(true)).to_bits(), serial.to_bits(), "{} threads", threads);
            }
        }
    }

    #[test]
    fn a_sum_adds_its_chunks_totals_in_order() {
        // Within a chunk, the terms add one after another; across chunks, their totals.
        let count = u32::try_from(2 * CHUNK + 3).expect("a few thousand terms");
        let terms: Vec<f64> = (1..=count).map(|index| 1.0 / f64::from(index)).collect();
        let chunks = terms
            .chunks(CHUNK)
            .map(|chunk| chunk.iter().fold(0.0, |a, b| a + b));
        let expected = chunks.fold(0.0, |a, b| a + b);
        assert_eq!(
            sum(terms.len(), |index| terms[index]).to_bits(),
            expected.to_bits()
        );
        assert_eq!(sum(0, |_| 1.0).to_bits(), 0.0_f64.to_bits());
    }
}
