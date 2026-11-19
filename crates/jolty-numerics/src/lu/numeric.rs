//! The numeric factorization of an ordered matrix's diagonal blocks, its refactorization
//! with the same pivots, and the solves with its factors.
//!
//! Each column is eliminated by the left-looking method of Gilbert and Peierls ("Sparse
//! partial pivoting in time proportional to arithmetic operations", SIAM Journal on
//! Scientific and Statistical Computing 9(5), 1988): a depth-first search through the
//! columns of `L` already found gives the column's pattern in an order that respects their
//! dependencies, and a sparse triangular solve in that order its values. Rows are indexed as
//! in the ordered matrix throughout, and each column's pivot row is recorded, so a
//! refactorization replays the same eliminations in the same order with neither search.

use rayon::prelude::*;

use super::ordering::RowsAbove;
use super::tree::Tree;
use super::{LinearError, Ordering, Scalar, Workspace};
use crate::reduce::{self, CHUNK};

/// No index: a row not yet a pivot, a row not yet visited.
///
/// Origin: sentinel `usize::MAX`, unreachable by any valid Rust `Vec` index.
const NONE: usize = usize::MAX;

/// The least fraction of the largest candidate in its column that a pivot is: a
/// factorization takes the ordering's diagonal entry as the pivot when it reaches this
/// fraction, and else the largest, and a refactorization keeps each pivot while it reaches
/// it. It bounds every entry of `L` by its inverse, ten, against a thousand for the
/// thousandth that KLU and SPICE take (threshold partial pivoting: Duff, Erisman and Reid,
/// *Direct Methods for Sparse Matrices*, 2nd ed., 2017; compare Davis and Palamadai
/// Natarajan, "Algorithm 907: KLU", *ACM Transactions on Mathematical Software* 37(3),
/// 2010, whose default is `0.001`), which keeps the rounding of the solutions close to
/// that of strict partial pivoting: over 20,000 of the tests' random circuits, the largest
/// backward error of a solution is 4.5e-15 with a tenth, against 7.7e-9 with a thousandth.
/// On the matrices of the example circuits, meshes and inverter chains measured, the
/// factors fill alike with either, but for one entry of one. The price is that a
/// refactorization keeps its pivots less often: in the transients of the examples' buck and
/// boost converters, one or two factorizations in a hundred search afresh, where with a
/// thousandth almost none does.
pub(super) const PIVOT_TOLERANCE: f64 = 0.1;

/// A triangular factor stored by columns.
#[derive(Clone)]
pub(super) struct Triangle<T> {
    /// Where each column's entries start, then their number.
    pub(super) starts: Vec<usize>,
    /// Each entry's index: a row of the ordered matrix in `L`, a position in `U`.
    pub(super) indices: Vec<usize>,
    /// Each entry's value.
    pub(super) values: Vec<T>,
}

impl<T: Scalar> Triangle<T> {
    pub(super) fn new(dimension: usize) -> Self {
        let mut starts = Vec::with_capacity(dimension + 1);
        starts.push(0);
        Self {
            starts,
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// The indices and values of `column`'s entries.
    pub(super) fn column(&self, column: usize) -> (&[usize], &[T]) {
        let range = self.starts[column]..self.starts[column + 1];
        (&self.indices[range.clone()], &self.values[range])
    }

    /// Keeps the first `columns` columns.
    pub(super) fn truncate(&mut self, columns: usize) {
        let end = self.starts[columns];
        self.starts.truncate(columns + 1);
        self.indices.truncate(end);
        self.values.truncate(end);
    }

    /// Makes room for `additional` more entries.
    pub(super) fn reserve(&mut self, additional: usize) -> Result<(), LinearError> {
        self.indices.try_reserve(additional)?;
        self.values.try_reserve(additional)?;
        Ok(())
    }

    /// Adds an entry to the column being built.
    pub(super) fn push(&mut self, index: usize, value: T) {
        self.indices.push(index);
        self.values.push(value);
    }

    /// Closes the column being built.
    pub(super) fn close(&mut self) {
        self.starts.push(self.indices.len());
    }
}

/// How a factorization went.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Factorization {
    /// It searched for every pivot, there being no pivots to keep.
    Searched,
    /// It kept every pivot of the last factorization.
    Refactored,
    /// It kept the pivots of the last factorization up to one that had become too small,
    /// and searched for the pivots from that one on.
    FellBack,
}

/// The factors `L U` of the rows of an ordered matrix's diagonal blocks, permuted within
/// each block by its pivots, and the workspace that finds them: of the blocks the ordering
/// dissects, by their trees, and of the others, column by column.
#[derive(Clone)]
pub(super) struct Numeric<T> {
    /// The factors of each dissected block, in the ordering's order.
    trees: Vec<Tree<T>>,
    /// The row of the ordered matrix that is each position's pivot.
    pivot_rows: Vec<usize>,
    /// The position whose pivot each row of the ordered matrix is, or `NONE`.
    positions: Vec<usize>,
    /// `L` below its unit diagonal.
    lower: Triangle<T>,
    /// `U` above its diagonal, each column's entries in the order of their eliminations.
    upper: Triangle<T>,
    /// The diagonal of `U`: the pivots.
    pivots: Vec<T>,
    /// Whether every position has its pivot, so that a refactorization can keep them.
    complete: bool,
    /// The column being eliminated, by row: zero outside it.
    work: Vec<T>,
    /// The column whose search last visited each row.
    marks: Vec<usize>,
    /// The rows being visited by the search, each with its next entry of `L` to follow.
    visits: Vec<(usize, usize)>,
    /// The rows of the column being eliminated, in an order whose reverse puts every row
    /// after each row whose elimination changes it.
    reach: Vec<usize>,
}

impl<T: Scalar> Numeric<T> {
    /// No factors yet of the matrices of `ordering`'s pattern.
    pub(super) fn new(ordering: &Ordering) -> Self {
        let dimension = ordering.factored().dimension();
        Self {
            trees: ordering.dissections().iter().map(Tree::new).collect(),
            pivot_rows: vec![NONE; dimension],
            positions: vec![NONE; dimension],
            lower: Triangle::new(dimension),
            upper: Triangle::new(dimension),
            pivots: vec![T::ZERO; dimension],
            complete: false,
            work: vec![T::ZERO; dimension],
            marks: vec![NONE; dimension],
            visits: Vec::with_capacity(dimension),
            reach: Vec::with_capacity(dimension),
        }
    }

    /// Forgets the pivots of the last factorization, so that the next searches for every
    /// pivot, as a new one's first does: with no pivot kept, a factorization reads nothing
    /// the last left but its allocations.
    pub(super) fn forget(&mut self) {
        self.complete = false;
        for tree in &mut self.trees {
            tree.forget();
        }
    }

    /// Whether the next factorization of any matrix keeps the same pivots, in the same
    /// patterns of `L` and `U`, as `other`'s would: both are complete, with the same pivot
    /// rows and the same patterns, or neither is, so that both search afresh. A
    /// refactorization reads nothing else: not the values of the factors, nor the
    /// workspace, which every elimination leaves zero.
    pub(super) fn pivots_alike(&self, other: &Self) -> bool {
        match (self.complete, other.complete) {
            (false, false) => true,
            (true, true) => {
                self.pivot_rows == other.pivot_rows
                    && self.lower.starts == other.lower.starts
                    && self.lower.indices == other.lower.indices
                    && self.upper.starts == other.upper.starts
                    && self.upper.indices == other.upper.indices
                    && self
                        .trees
                        .iter()
                        .zip(&other.trees)
                        .all(|(one, another)| one.pivots_alike(another))
            }
            _ => false,
        }
    }

    /// The number of columns the last factorization delayed from a node of a dissected
    /// block to its parent.
    pub(super) fn delays(&self) -> usize {
        self.trees.iter().map(Tree::delays).sum()
    }

    /// The number of entries of `L` below its diagonal and of `U` on and above it.
    pub(super) fn entries(&self) -> usize {
        let dissected: usize = self.trees.iter().map(Tree::entries).sum();
        let positions = self.pivot_rows.iter().filter(|&&row| row != NONE).count();
        self.lower.indices.len() + self.upper.indices.len() + positions + dissected
    }

    /// Factors the matrix of `ordering`'s pattern whose values are `values`, keeping the
    /// last factorization's pivots while each passes, and searching for the pivots from the
    /// first that does not: in each dissected block, in each node whose pivot does not and
    /// in the nodes above it. Returns how it went.
    pub(super) fn factor(
        &mut self,
        ordering: &Ordering,
        values: &[T],
        threads: usize,
    ) -> Result<Factorization, LinearError> {
        let afresh = !self.complete;
        let (first, mut went) = if afresh {
            (Some(0), Factorization::Searched)
        } else {
            match self.refactor(ordering, values) {
                None => (None, Factorization::Refactored),
                Some(first) => (Some(first), Factorization::FellBack),
            }
        };
        self.complete = false;
        if let Some(first) = first {
            if afresh {
                self.positions.fill(NONE);
            } else {
                for &row in &self.pivot_rows[first..] {
                    if row != NONE {
                        self.positions[row] = NONE;
                    }
                }
            }
            self.lower.truncate(first);
            self.upper.truncate(first);
            self.marks.fill(NONE);
            for (block, range) in ordering.block_starts().windows(2).enumerate() {
                let dissected = ordering.dissection(block).is_some();
                for column in range[0].max(first)..range[1] {
                    if dissected {
                        self.pivot_rows[column] = NONE;
                        self.lower.close();
                        self.upper.close();
                    } else {
                        self.eliminate(ordering, values, column)?;
                    }
                }
            }
        }
        for (tree, dissection) in self.trees.iter_mut().zip(ordering.dissections()) {
            if afresh {
                tree.forget();
            }
            let searched =
                tree.factor(dissection, values, threads)
                    .map_err(|error| match error {
                        LinearError::Singular { columns } => LinearError::Singular {
                            columns: columns
                                .into_iter()
                                .map(|position| ordering.columns()[position])
                                .collect(),
                        },
                        LinearError::TooLarge => LinearError::TooLarge,
                    })?;
            if searched && !afresh {
                went = Factorization::FellBack;
            }
        }
        debug_assert!(self.trees.iter().all(Tree::is_complete));
        self.complete = true;
        Ok(went)
    }

    /// Refactors with the pivots and patterns of the last factorization. Returns `None`
    /// when every pivot reaches its tolerance, and else the first position whose pivot does
    /// not, the factors before it found.
    fn refactor(&mut self, ordering: &Ordering, values: &[T]) -> Option<usize> {
        let Self {
            work,
            upper,
            lower,
            pivot_rows,
            pivots,
            ..
        } = self;
        for (block, range) in ordering.block_starts().windows(2).enumerate() {
            if ordering.dissection(block).is_some() {
                continue;
            }
            for column in range[0]..range[1] {
                for &(row, slot) in ordering.within(column) {
                    work[row] = values[slot];
                }
                for entry in upper.starts[column]..upper.starts[column + 1] {
                    let position = upper.indices[entry];
                    let row = pivot_rows[position];
                    let multiplier = work[row];
                    work[row] = T::ZERO;
                    upper.values[entry] = multiplier;
                    let (indices, factors) = lower.column(position);
                    for (&index, &factor) in indices.iter().zip(factors) {
                        work[index] -= factor * multiplier;
                    }
                }
                let row = pivot_rows[column];
                let pivot = work[row];
                work[row] = T::ZERO;
                // The largest entry below the pivot, as each is divided by it.
                let mut largest = 0.0;
                let below = lower.starts[column]..lower.starts[column + 1];
                for (&index, value) in lower.indices[below.clone()]
                    .iter()
                    .zip(&mut lower.values[below])
                {
                    let entry = work[index];
                    largest = f64::max(largest, entry.magnitude());
                    *value = entry / pivot;
                    work[index] = T::ZERO;
                }
                let magnitude = pivot.magnitude();
                if !(magnitude > 0.0 && magnitude >= PIVOT_TOLERANCE * largest) {
                    return Some(column);
                }
                pivots[column] = pivot;
            }
        }
        None
    }

    /// Eliminates `column`, choosing its pivot, after every column before it.
    fn eliminate(
        &mut self,
        ordering: &Ordering,
        values: &[T],
        column: usize,
    ) -> Result<(), LinearError> {
        self.find_reach(ordering, column);
        for &(row, slot) in ordering.within(column) {
            self.work[row] = values[slot];
        }
        for &row in self.reach.iter().rev() {
            let position = self.positions[row];
            if position == NONE {
                continue;
            }
            let multiplier = self.work[row];
            let (indices, factors) = self.lower.column(position);
            for (&index, &factor) in indices.iter().zip(factors) {
                self.work[index] -= factor * multiplier;
            }
        }
        if let Err(error) = self
            .upper
            .reserve(self.reach.len())
            .and_then(|()| self.lower.reserve(self.reach.len()))
        {
            self.clear_reach();
            return Err(error);
        }
        let (mut largest, mut chosen) = (0.0, NONE);
        for &row in self.reach.iter().rev() {
            let position = self.positions[row];
            if position == NONE {
                let magnitude = self.work[row].magnitude();
                if magnitude > largest {
                    largest = magnitude;
                    chosen = row;
                }
            } else {
                self.upper.push(position, self.work[row]);
                self.work[row] = T::ZERO;
            }
        }
        // The ordering expects the row it matched with this column as the pivot.
        if chosen != NONE
            && self.positions[column] == NONE
            && self.work[column].magnitude() >= PIVOT_TOLERANCE * largest
        {
            chosen = column;
        }
        if chosen == NONE {
            self.clear_reach();
            return Err(LinearError::Singular {
                columns: vec![ordering.columns()[column]],
            });
        }
        let pivot = self.work[chosen];
        self.work[chosen] = T::ZERO;
        self.pivot_rows[column] = chosen;
        self.positions[chosen] = column;
        self.pivots[column] = pivot;
        for &row in &self.reach {
            if self.positions[row] == NONE {
                self.lower.push(row, self.work[row] / pivot);
                self.work[row] = T::ZERO;
            }
        }
        self.lower.close();
        self.upper.close();
        Ok(())
    }

    /// Leaves the column being eliminated zero.
    fn clear_reach(&mut self) {
        for &row in &self.reach {
            self.work[row] = T::ZERO;
        }
    }

    /// Finds the rows of `column` of `L U`: those of the matrix's column and every row an
    /// entry of `L` reaches from them, in `reach`.
    fn find_reach(&mut self, ordering: &Ordering, column: usize) {
        self.reach.clear();
        for &(start, _) in ordering.within(column) {
            if self.marks[start] == column {
                continue;
            }
            self.marks[start] = column;
            self.visits.push((start, self.first_entry(start)));
            while let Some(visit) = self.visits.last_mut() {
                let (row, entry) = *visit;
                let position = self.positions[row];
                if position != NONE && entry < self.lower.starts[position + 1] {
                    visit.1 += 1;
                    let next = self.lower.indices[entry];
                    if self.marks[next] != column {
                        self.marks[next] = column;
                        let first = self.first_entry(next);
                        self.visits.push((next, first));
                    }
                } else {
                    self.visits.pop();
                    self.reach.push(row);
                }
            }
        }
    }

    /// Where the entries of `L` to follow from `row` start: its pivot's column's, if it is
    /// a pivot.
    fn first_entry(&self, row: usize) -> usize {
        match self.positions[row] {
            NONE => 0,
            position => self.lower.starts[position],
        }
    }

    /// Solves `A x = rhs` and leaves `x` in `rhs`, for the matrix of `ordering`'s pattern
    /// with `values` whose factors these are, writing to `workspace` as it goes, its
    /// dissected blocks spread over `threads` threads as [`super::SparseLu::spread`] says.
    pub(super) fn solve(
        &self,
        (ordering, values): (&Ordering, &[T]),
        rhs: &mut [T],
        workspace: &mut Workspace<T>,
        threads: usize,
    ) {
        let Workspace {
            by_row,
            by_position,
            dissected: scratch,
            ..
        } = workspace;
        scratch.resize_with(self.trees.len(), Vec::new);
        gather(by_row, rhs, ordering.rows(), threads);
        for (block, range) in ordering.block_starts().windows(2).enumerate().rev() {
            let positions = range[0]..range[1];
            if let Some(dissection) = ordering.dissection(block) {
                let index = ordering.dissected_index(block);
                self.trees[index].solve(
                    dissection,
                    (by_row, &mut by_position[positions.clone()]),
                    &mut scratch[index],
                    (false, threads),
                );
            } else {
                for position in positions.clone() {
                    let value = by_row[self.pivot_rows[position]];
                    by_position[position] = value;
                    let (indices, factors) = self.lower.column(position);
                    for (&index, &factor) in indices.iter().zip(factors) {
                        by_row[index] -= factor * value;
                    }
                }
                for position in positions.clone().rev() {
                    let value = by_position[position] / self.pivots[position];
                    by_position[position] = value;
                    let (indices, factors) = self.upper.column(position);
                    for (&index, &factor) in indices.iter().zip(factors) {
                        by_position[index] -= factor * value;
                    }
                }
            }
            take_in_above(
                ordering.across(block),
                (values, by_position),
                (by_row, 0),
                threads,
            );
        }
        gather(rhs, by_position, ordering.column_positions(), threads);
    }

    /// Solves `Aᵀ x = rhs` and leaves `x` in `rhs`, as [`Numeric::solve`] does `A x = rhs`.
    pub(super) fn solve_transpose(
        &self,
        (ordering, values): (&Ordering, &[T]),
        rhs: &mut [T],
        workspace: &mut Workspace<T>,
        threads: usize,
    ) {
        let Workspace {
            by_row,
            by_position,
            dissected: scratch,
            ..
        } = workspace;
        scratch.resize_with(self.trees.len(), Vec::new);
        gather(by_position, rhs, ordering.columns(), threads);
        for (block, range) in ordering.block_starts().windows(2).enumerate() {
            let positions = range[0]..range[1];
            if let Some(dissection) = ordering.dissection(block) {
                let taken = &mut by_position[positions.clone()];
                let take = |first: usize, chunk: &mut [T]| {
                    for (offset, value) in chunk.iter_mut().enumerate() {
                        let above = ordering.above(positions.start + first + offset);
                        *value = take_in(above, *value, (values, by_row), threads);
                    }
                };
                // Each position takes in the rows above it alone, so the positions spread.
                if threads < 2 || taken.len() <= CHUNK {
                    take(0, taken);
                } else {
                    taken
                        .par_chunks_mut(CHUNK)
                        .enumerate()
                        .for_each(|(index, chunk)| take(index * CHUNK, chunk));
                }
                let index = ordering.dissected_index(block);
                self.trees[index].solve(
                    dissection,
                    (by_position, &mut by_row[positions]),
                    &mut scratch[index],
                    (true, threads),
                );
                continue;
            }
            for position in positions.clone() {
                let above = ordering.above(position);
                let mut value = take_in(above, by_position[position], (values, by_row), threads);
                let (indices, factors) = self.upper.column(position);
                for (&index, &factor) in indices.iter().zip(factors) {
                    value -= factor * by_position[index];
                }
                by_position[position] = value / self.pivots[position];
            }
            for position in positions.rev() {
                let mut value = by_position[position];
                let (indices, factors) = self.lower.column(position);
                for (&index, &factor) in indices.iter().zip(factors) {
                    value -= factor * by_row[index];
                }
                by_row[self.pivot_rows[position]] = value;
            }
        }
        gather(rhs, by_row, ordering.row_positions(), threads);
    }
}

/// Has each of `rows`, rows above a diagonal block, take in the block's columns, whose
/// solution `by_position` holds, through its entries' `values`: of `by_row`, the values of
/// the rows from `first` on. Each row takes them in in increasing position, as
/// [`take_in`] does, the order in which a pass over the columns would take them in; the
/// rows are independent, so for two `threads` or more they spread over the current rayon
/// pool's threads in runs of [`CHUNK`] rows, the same bits either way.
///
/// The rows taken in on the calling thread are inlined where it is called, so that a small
/// circuit's solve, a few rows above each of its blocks, pays nothing for spreading.
#[expect(
    clippy::inline_always,
    reason = "called once a block; on the buck's solves, a call cost 28 million instructions where the loop inlined costs 21"
)]
#[inline(always)]
fn take_in_above<T: Scalar>(
    rows: RowsAbove<'_>,
    (values, by_position): (&[T], &[T]),
    (by_row, first): (&mut [T], usize),
    threads: usize,
) {
    if threads < 2 || rows.len() <= CHUNK {
        for (row, entries) in rows.iter() {
            let row = &mut by_row[row - first];
            *row = take_in(entries, *row, (values, by_position), threads);
        }
        return;
    }
    spread_above(rows, (values, by_position), (by_row, first), threads);
}

/// [`take_in_above`] of more than [`CHUNK`] rows on two `threads` or more: each half of
/// them as a task of its own. Never inlined, so that the rows taken in on the calling
/// thread stay a loop small enough to inline where they are called.
#[inline(never)]
fn spread_above<T: Scalar>(
    rows: RowsAbove<'_>,
    (values, by_position): (&[T], &[T]),
    (by_row, first): (&mut [T], usize),
    threads: usize,
) {
    // Halved at a whole number of runs, the later rows at and after the first of theirs.
    let half = rows.len().div_ceil(CHUNK) / 2 * CHUNK;
    let (before, after) = rows.split_at(half);
    let at = rows.row(half);
    let (earlier, later) = by_row.split_at_mut(at - first);
    rayon::join(
        || take_in_above(before, (values, by_position), (earlier, first), threads),
        || take_in_above(after, (values, by_position), (later, at), threads),
    );
}

/// `value` less each of `entries`' `values` times the value `solved` holds at its index, in
/// the entries' order: one after another for no more than [`CHUNK`] entries, and else by
/// [`reduce::fold`]'s chunks of them, which spread over the current rayon pool's threads
/// for two `threads` or more, so that a long row, such as a supply's joined to every
/// device, is no single sum on one thread.
#[inline]
fn take_in<T: Scalar>(
    entries: &[(usize, usize)],
    value: T,
    (values, solved): (&[T], &[T]),
    threads: usize,
) -> T {
    reduce::fold(
        entries,
        (value, T::ZERO),
        |value, &(at, slot)| value - values[slot] * solved[at],
        |total, chunk| total + chunk,
        threads >= 2,
    )
}

/// Sets each of `into` to the value of `from` at its index in `indices`, in chunks of
/// [`CHUNK`] spread over the current rayon pool's threads when `threads` is two or more and
/// the copy spans more than one chunk.
fn gather<T: Scalar>(into: &mut [T], from: &[T], indices: &[usize], threads: usize) {
    if threads < 2 || into.len().min(indices.len()) <= CHUNK {
        for (value, &index) in into.iter_mut().zip(indices) {
            *value = from[index];
        }
        return;
    }
    into.par_chunks_mut(CHUNK)
        .zip(indices.par_chunks(CHUNK))
        .for_each(|(values, indices)| {
            for (value, &index) in values.iter_mut().zip(indices) {
                *value = from[index];
            }
        });
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::PatternBuilder;

    /// The values by slot of `ordering`'s dense 3 × 3 pattern that make its ordered matrix
    /// `matrix`.
    fn ordered(ordering: &Ordering, matrix: [[f64; 3]; 3]) -> Vec<f64> {
        let at = |order: &[usize], wanted: usize| {
            order
                .iter()
                .position(|&index| index == wanted)
                .expect("a permutation")
        };
        ordering
            .pattern()
            .positions()
            .map(|(row, column)| matrix[at(ordering.rows(), row)][at(ordering.columns(), column)])
            .collect()
    }

    #[test]
    fn a_refactorization_keeps_the_columns_before_a_pivot_too_small() {
        let mut builder = PatternBuilder::new();
        for row in 0..3 {
            for column in 0..3 {
                builder.entry(row, column);
            }
        }
        let ordering = Ordering::new(Arc::new(builder.finish(3))).expect("fits");
        assert_eq!(ordering.blocks(), 1);
        let first = ordered(
            &ordering,
            [[4.0, 1.0, 1.0], [1.0, 4.0, 1.0], [1.0, 1.0, 4.0]],
        );
        // The first pivot of the second matrix still holds, and its second, 10⁻¹⁴, is far
        // below the entry under it.
        let second = ordered(
            &ordering,
            [[4.0, 0.0, 0.0], [0.0, 1e-14, 1.0], [0.0, 1.0, 1e-14]],
        );
        let mut numeric = Numeric::new(&ordering);
        assert_eq!(
            numeric.factor(&ordering, &first, 1),
            Ok(Factorization::Searched)
        );
        assert_eq!(numeric.pivot_rows, [0, 1, 2]);
        assert_eq!(numeric.refactor(&ordering, &first), None);
        assert_eq!(numeric.refactor(&ordering, &second), Some(1));
        assert_eq!(
            numeric.factor(&ordering, &second, 1),
            Ok(Factorization::FellBack)
        );
        assert_eq!(numeric.pivot_rows, [0, 2, 1]);
        // The factors solve the second matrix's system: its residual is a rounding.
        let wanted = [1.0, -2.0, 3.0];
        let mut x = wanted;
        let mut workspace = Workspace::new(3);
        numeric.solve((&ordering, &second), &mut x, &mut workspace, 1);
        let mut residual = wanted;
        for ((row, column), value) in ordering.pattern().positions().zip(&second) {
            residual[row] -= value * x[column];
        }
        assert!(
            residual.iter().all(|value| value.abs() < 1e-15),
            "{residual:?}"
        );
    }
}
