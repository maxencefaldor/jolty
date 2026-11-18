//! Sums of many terms in one fixed order, whether one thread adds them or many.
//!
//! [`sum`] adds its terms in chunks of [`CHUNK`] consecutive terms, each chunk from its first
//! term to its last, then the chunks' totals in their order. [`sum_in_parallel`] adds the
//! same chunks, each on whichever thread of the current rayon pool takes it, and their
//! totals in the same order, so the two give the same bits, whatever the threads. [`fold`]
//! folds terms of any kind into a value in the same chunks, the first chunk into the value
//! and each later one from zero, on the calling thread or spread, so that a long sum spreads
//! and one of a chunk or less is its terms one after another. Never rayon's own `sum`,
//! whose grouping follows the threads' work.

use rayon::prelude::*;

/// The terms a chunk adds before its total joins the others'.
///
/// Origin: jolty's own power-of-two deterministic-reduction chunk size (`2^10 = 1024` `f64`
/// values, 8,192 bytes), fitting within a typical L1 data cache while amortizing rayon
/// task-scheduling overhead across threads.
pub const CHUNK: usize = 1024;

/// The sum of the `count` terms `term` gives for the indices from 0, in chunks of
/// [`CHUNK`].
pub fn sum(count: usize, term: impl Fn(usize) -> f64) -> f64 {
    (0..count.div_ceil(CHUNK)).fold(0.0, |total, chunk| total + chunk_sum(chunk, count, &term))
}

/// The sum [`sum`] gives, its chunks added on the threads of the current rayon pool, at
/// least `least` of them a task.
pub fn sum_in_parallel(count: usize, term: impl Fn(usize) -> f64 + Sync, least: usize) -> f64 {
    let totals: Vec<f64> = (0..count.div_ceil(CHUNK))
        .into_par_iter()
        .with_min_len(least.max(1))
        .map(|chunk| chunk_sum(chunk, count, &term))
        .collect();
    totals.into_iter().fold(0.0, |total, chunk| total + chunk)
}

/// The sum of the terms of the chunk `chunk` of `count` terms, from its first to its last.
fn chunk_sum(chunk: usize, count: usize, term: &impl Fn(usize) -> f64) -> f64 {
    let start = chunk * CHUNK;
    (start..count.min(start + CHUNK)).fold(0.0, |total, index| total + term(index))
}

/// Folds `items` into `start`, one after another, when they fill one chunk of [`CHUNK`] or
/// less; and else in chunks of [`CHUNK`] consecutive items, the first folded into `start`
/// and each later one into `zero`, each from its first item to its last, then the later
/// chunks' totals joined into the first's, in their order. `fold` folds one item into a
/// total, and `join` one chunk's total into another's.
///
/// The chunks are the same however many threads take them, so the result is the same bits
/// on the calling thread alone or, `spread`, with the chunks spread over the threads of the
/// current rayon pool; and for a fold of no more than [`CHUNK`] items it is their fold one
/// after another, so that a long sum spreads where a short one stays as it was.
///
/// A short fold is inlined where it is called, as the loop it is, so that the many short
/// sums of a small circuit's solve pay nothing for the long ones'.
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

/// [`fold`] of more than [`CHUNK`] items, by its chunks. Never inlined, so that a short fold
/// stays the loop it is where it is called.
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
        /// Summed on 1 to 8 threads, in tasks of 1 to 3 chunks at least, the terms give the
        /// bits one thread gives.
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
        /// A fold of up to a chunk of terms takes them one after another into its start; a
        /// longer one takes each chunk from its first term to its last, the first into the
        /// start and each later one from zero, then their totals in order: the same bits on
        /// the calling thread and spread over 1, 2, 3 and 8 threads.
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
