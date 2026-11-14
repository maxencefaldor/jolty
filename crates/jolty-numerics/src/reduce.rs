//! Sums of many terms in one fixed order, whether one thread adds them or many.
//!
//! [`sum`] adds its terms in chunks of [`CHUNK`] consecutive terms, each chunk from its first
//! term to its last, then the chunks' totals in their order. [`sum_in_parallel`] adds the
//! same chunks, each on whichever thread of the current rayon pool takes it, and their
//! totals in the same order, so the two give the same bits, whatever the threads. Never
//! rayon's own `sum`, whose grouping follows the threads' work.

use rayon::prelude::*;

/// The terms a chunk adds before its total joins the others'.
pub const CHUNK: usize = 1024;

/// The sum of the `count` terms `term` gives for the indices from 0, in chunks of
/// [`CHUNK`].
pub fn sum(count: usize, term: impl Fn(usize) -> f64) -> f64 {
    (0..count.div_ceil(CHUNK)).fold(0.0, |total, chunk| total + chunk_sum(chunk, count, &term))
}

/// The sum [`sum`] gives, its chunks added on the threads of the current rayon pool.
pub fn sum_in_parallel(count: usize, term: impl Fn(usize) -> f64 + Sync) -> f64 {
    let totals: Vec<f64> = (0..count.div_ceil(CHUNK))
        .into_par_iter()
        .map(|chunk| chunk_sum(chunk, count, &term))
        .collect();
    totals.into_iter().fold(0.0, |total, chunk| total + chunk)
}

/// The sum of the terms of the chunk `chunk` of `count` terms, from its first to its last.
fn chunk_sum(chunk: usize, count: usize, term: &impl Fn(usize) -> f64) -> f64 {
    let start = chunk * CHUNK;
    (start..count.min(start + CHUNK)).fold(0.0, |total, index| total + term(index))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// Summed on 1 to 8 threads, the terms give the bits one thread gives.
        #[test]
        fn a_sum_on_many_threads_is_the_sum_on_one_bit_for_bit(
            terms in proptest::collection::vec(-1e3..1e3f64, 0..5000),
            threads in 1..9usize,
        ) {
            let serial = sum(terms.len(), |index| terms[index]);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("a pool");
            let parallel = pool.install(|| sum_in_parallel(terms.len(), |index| terms[index]));
            prop_assert_eq!(serial.to_bits(), parallel.to_bits());
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
