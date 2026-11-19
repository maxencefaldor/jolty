//! The numeric factorization of an ordered matrix's diagonal blocks, its refactorization
//! with the same pivots, and the solves with its factors.
//!
//! Each column is eliminated by the left-looking method of Gilbert and Peierls ("Sparse
//! partial pivoting in time proportional to arithmetic operations", SIAM Journal on
//! Scientific and Statistical Computing 9(5), 1988). A depth-first search through existing
//! columns of `L` yields the column's sparsity pattern in dependency order, and a sparse
//! triangular solve computes the entries. Rows stay indexed in ordered-matrix coordinates
//! and each column records its pivot row, so refactorization replays the same eliminations
//! without repeating either search.

use rayon::prelude::*;

use super::ordering::RowsAbove;
use super::tree::Tree;
use super::{LinearError, NONE, Ordering, Scalar, Workspace};
use crate::reduce::{self, CHUNK};

/// Minimum fraction of the largest candidate in a column required to accept a pivot during
/// threshold partial pivoting and refactorization.
///
/// Source: Duff, Erisman, and Reid (*Direct Methods for Sparse Matrices*, 2nd ed., 2017)
/// and Davis and Palamadai Natarajan ("Algorithm 907: KLU", *ACM TOMS* 37(3), 2010).
/// Bounding `L` entries by `1 / PIVOT_TOLERANCE = 10` (versus `1000` at the `0.001` KLU
/// and SPICE default) keeps max backward error at `4.5e-15` versus `7.7e-9` across 20,000
/// random test circuits. Fill stays virtually identical, with a 1 to 2% fallback rate on
/// switching-converter transients.
pub(super) const PIVOT_TOLERANCE: f64 = 0.1;

/// Triangular factor stored in compressed sparse column format.
#[derive(Clone)]
pub(super) struct Triangle<T> {
    /// Column start offsets in `indices` and `values`, followed by total length.
    starts: Vec<usize>,
    /// Row index (for `L`) or step position (for `U`) of each stored entry.
    indices: Vec<usize>,
    /// Scalar value of each stored entry.
    values: Vec<T>,
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

    /// Returns the total number of stored entries.
    pub(super) fn entries(&self) -> usize {
        self.indices.len()
    }

    /// Returns whether `self` and `other` have identical sparsity patterns.
    pub(super) fn pattern_eq(&self, other: &Self) -> bool {
        self.starts == other.starts && self.indices == other.indices
    }

    /// Returns the index and value slices for `column`.
    fn column(&self, column: usize) -> (&[usize], &[T]) {
        let range = self.starts[column]..self.starts[column + 1];
        (&self.indices[range.clone()], &self.values[range])
    }

    /// Subtracts `multiplier` times `column` from `target`.
    #[inline]
    pub(super) fn subtract(&self, column: usize, multiplier: T, target: &mut [T]) {
        let (indices, factors) = self.column(column);
        for (&index, &factor) in indices.iter().zip(factors) {
            target[index] -= factor * multiplier;
        }
    }

    /// Returns `value` minus the dot product of `column` with `source`.
    #[inline]
    pub(super) fn subtract_dot(&self, column: usize, mut value: T, source: &[T]) -> T {
        let (indices, factors) = self.column(column);
        for (&index, &factor) in indices.iter().zip(factors) {
            value -= factor * source[index];
        }
        value
    }

    /// Keeps the first `columns` columns.
    pub(super) fn truncate(&mut self, columns: usize) {
        let end = self.starts[columns];
        self.starts.truncate(columns + 1);
        self.indices.truncate(end);
        self.values.truncate(end);
    }

    /// Makes room for `additional` more entries.
    fn reserve(&mut self, additional: usize) -> Result<(), LinearError> {
        self.indices.try_reserve(additional)?;
        self.values.try_reserve(additional)?;
        Ok(())
    }

    /// Adds an entry to the column being built.
    fn push(&mut self, index: usize, value: T) {
        self.indices.push(index);
        self.values.push(value);
    }

    /// Closes the column being built.
    fn close(&mut self) {
        self.starts.push(self.indices.len());
    }

    /// Moves the pivoted entries of `reach` from `work` into `self` in reverse postorder.
    fn drain_pivoted(&mut self, reach: &[usize], positions: &[usize], work: &mut [T]) {
        for &row in reach.iter().rev() {
            let earlier = positions[row];
            if earlier != NONE {
                self.push(earlier, work[row]);
                work[row] = T::ZERO;
            }
        }
    }

    /// Scatters `entries` from `values` into `work` and replays `column` of `self` through
    /// `lower` and `pivot_rows`, overwriting `self.values` and clearing each replayed pivot
    /// row in `work`.
    pub(super) fn replay(
        &mut self,
        (entries, values): (&[(usize, usize)], &[T]),
        (pivot_rows, lower): (&[usize], &Self),
        work: &mut [T],
        column: usize,
    ) {
        for &(row, slot) in entries {
            work[row] = values[slot];
        }
        for entry in self.starts[column]..self.starts[column + 1] {
            let earlier = self.indices[entry];
            let row = pivot_rows[earlier];
            let multiplier = work[row];
            work[row] = T::ZERO;
            self.values[entry] = multiplier;
            lower.subtract(earlier, multiplier, work);
        }
    }
}

/// Column-by-column sparse `L U` factors and Gilbert-Peierls elimination workspace, shared
/// by full diagonal blocks ([`Numeric`]) and dissected interiors.
#[derive(Clone)]
pub(super) struct ColumnFactors<T> {
    /// The row pivoted at each step or position.
    pub(super) pivot_rows: Vec<usize>,
    /// The step or position whose pivot each row is, or [`NONE`].
    pub(super) positions: Vec<usize>,
    /// Lower-triangular factor `L` below its unit diagonal.
    pub(super) lower: Triangle<T>,
    /// Upper-triangular factor `U` above its diagonal, stored in elimination order per column.
    pub(super) upper: Triangle<T>,
    /// The diagonal of `U`: the pivots.
    pub(super) pivots: Vec<T>,
    /// The column being eliminated, by row: zero outside it.
    pub(super) work: Vec<T>,
    /// The stamp of the search that last visited each row.
    marks: Vec<usize>,
    /// The rows being visited by the search, each with its next entry of `L` to follow.
    visits: Vec<(usize, usize)>,
    /// Active rows of the column being eliminated in postorder, so reversing yields
    /// topological elimination order.
    reach: Vec<usize>,
}

impl<T: Scalar> ColumnFactors<T> {
    /// Allocates empty factors and workspace for `dimension` rows and columns.
    pub(super) fn new(dimension: usize) -> Self {
        Self {
            pivot_rows: vec![NONE; dimension],
            positions: vec![NONE; dimension],
            lower: Triangle::new(dimension),
            upper: Triangle::new(dimension),
            pivots: vec![T::ZERO; dimension],
            work: vec![T::ZERO; dimension],
            marks: vec![NONE; dimension],
            visits: Vec::with_capacity(dimension),
            reach: Vec::with_capacity(dimension),
        }
    }

    /// Resets the factors and workspace for a fresh search over `size` frame rows.
    pub(super) fn reset(&mut self, size: usize) {
        self.positions.clear();
        self.positions.resize(size, NONE);
        self.work.clear();
        self.work.resize(size, T::ZERO);
        self.marks.clear();
        self.marks.resize(size, NONE);
        self.pivot_rows.clear();
        self.pivots.clear();
        self.lower.truncate(0);
        self.upper.truncate(0);
    }

    /// Returns whether `self` and `other` hold identical pivot rows and triangular sparsity
    /// patterns.
    pub(super) fn patterns_alike(&self, other: &Self) -> bool {
        self.pivot_rows == other.pivot_rows
            && self.lower.pattern_eq(&other.lower)
            && self.upper.pattern_eq(&other.upper)
    }

    /// Returns the total entry count in strict lower `L` plus diagonal and upper `U`.
    pub(super) fn entries(&self) -> usize {
        let pivoted = self.pivot_rows.iter().filter(|&&row| row != NONE).count();
        self.lower.entries() + self.upper.entries() + pivoted
    }

    /// Leaves the column being eliminated zero.
    fn clear_reach(&mut self) {
        for &row in &self.reach {
            self.work[row] = T::ZERO;
        }
    }

    /// Finds the reach of `entries` under `stamp`, scatters `values` into `work`, and
    /// eliminates every earlier pivot in reverse postorder.
    pub(super) fn scatter_and_solve(
        &mut self,
        entries: &[(usize, usize)],
        values: &[T],
        stamp: usize,
    ) {
        self.find_reach(entries, stamp);
        for &(row, slot) in entries {
            self.work[row] = values[slot];
        }
        for &row in self.reach.iter().rev() {
            let position = self.positions[row];
            if position != NONE {
                let multiplier = self.work[row];
                self.lower.subtract(position, multiplier, &mut self.work);
            }
        }
    }

    /// Drains the pivoted rows of `reach` from `work` into `target` in reverse postorder
    /// and closes the column.
    pub(super) fn store_upper_into(&mut self, target: &mut Triangle<T>) -> Result<(), LinearError> {
        if let Err(error) = target.reserve(self.reach.len()) {
            self.clear_reach();
            return Err(error);
        }
        target.drain_pivoted(&self.reach, &self.positions, &mut self.work);
        target.close();
        Ok(())
    }

    /// Eliminates `column` (with entries `entries` in `values`) into `step`, choosing a
    /// pivot among rows below `allowed`. Returns `Ok(Some((pivot_row, pivot)))` when a
    /// pivot meets [`PIVOT_TOLERANCE`], or `Ok(None)` (leaving `work` zero) when none does.
    pub(super) fn eliminate_column(
        &mut self,
        entries: &[(usize, usize)],
        values: &[T],
        (column, step, allowed): (usize, usize, usize),
    ) -> Result<Option<(usize, T)>, LinearError> {
        self.scatter_and_solve(entries, values, column);
        let (mut largest, mut best, mut chosen) = (0.0, 0.0, NONE);
        for &row in self.reach.iter().rev() {
            if self.positions[row] == NONE {
                let magnitude = self.work[row].magnitude();
                largest = f64::max(largest, magnitude);
                if row < allowed && magnitude > best {
                    (best, chosen) = (magnitude, row);
                }
            }
        }
        let least = PIVOT_TOLERANCE * largest;
        // The ordering expects the row it matched with this column as the pivot.
        let diagonal = self.work[column].magnitude();
        let chosen = if self.positions[column] == NONE && diagonal > 0.0 && diagonal >= least {
            column
        } else if chosen != NONE && best >= least {
            chosen
        } else {
            self.clear_reach();
            return Ok(None);
        };
        if let Err(error) = self
            .upper
            .reserve(self.reach.len())
            .and_then(|()| self.lower.reserve(self.reach.len()))
        {
            self.clear_reach();
            return Err(error);
        }
        self.upper
            .drain_pivoted(&self.reach, &self.positions, &mut self.work);
        let pivot = self.work[chosen];
        self.work[chosen] = T::ZERO;
        self.positions[chosen] = step;
        for &row in &self.reach {
            if self.positions[row] == NONE {
                self.lower.push(row, self.work[row] / pivot);
                self.work[row] = T::ZERO;
            }
        }
        self.lower.close();
        self.upper.close();
        Ok(Some((chosen, pivot)))
    }

    /// Replays one already-pivoted column (`step`): scatters `entries`, eliminates against
    /// `upper` and `lower`, scales `lower`'s column by the pivot while clearing `work`, and
    /// returns `Some(pivot)` when it still meets [`PIVOT_TOLERANCE`].
    pub(super) fn refactor_column(
        &mut self,
        entries: &[(usize, usize)],
        values: &[T],
        step: usize,
    ) -> Option<T> {
        self.upper.replay(
            (entries, values),
            (&self.pivot_rows, &self.lower),
            &mut self.work,
            step,
        );
        let row = self.pivot_rows[step];
        let pivot = self.work[row];
        self.work[row] = T::ZERO;
        // The largest entry below the pivot, as each is divided by it.
        let mut largest = 0.0;
        let below = self.lower.starts[step]..self.lower.starts[step + 1];
        for (&index, value) in self.lower.indices[below.clone()]
            .iter()
            .zip(&mut self.lower.values[below])
        {
            let entry = self.work[index];
            largest = f64::max(largest, entry.magnitude());
            *value = entry / pivot;
            self.work[index] = T::ZERO;
        }
        let magnitude = pivot.magnitude();
        (magnitude > 0.0 && magnitude >= PIVOT_TOLERANCE * largest).then_some(pivot)
    }

    /// Finds the rows reached from `entries` through `lower` under visit `stamp`, storing
    /// them in `reach` in postorder.
    fn find_reach(&mut self, entries: &[(usize, usize)], stamp: usize) {
        self.reach.clear();
        for &(start, _) in entries {
            if self.marks[start] == stamp {
                continue;
            }
            self.marks[start] = stamp;
            self.visits.push((start, self.first_entry(start)));
            while let Some(visit) = self.visits.last_mut() {
                let (row, entry) = *visit;
                let position = self.positions[row];
                if position != NONE && entry < self.lower.starts[position + 1] {
                    visit.1 += 1;
                    let next = self.lower.indices[entry];
                    if self.marks[next] != stamp {
                        self.marks[next] = stamp;
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

    /// Returns the starting offset in `lower` for `row`'s pivot column, or `0` if unpivoted.
    fn first_entry(&self, row: usize) -> usize {
        match self.positions[row] {
            NONE => 0,
            position => self.lower.starts[position],
        }
    }
}

/// Strategy taken by a completed numeric factorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Factorization {
    /// Full pivot search performed because no reusable pivots were cached.
    Searched,
    /// Complete refactorization reusing every cached pivot.
    Refactored,
    /// Partial refactorization that fell back to pivot search after a cached pivot violated
    /// threshold tolerance.
    FellBack,
}

/// Numeric `L U` factors and workspaces for all diagonal blocks of an ordered matrix, using
/// multifrontal trees for dissected blocks and left-looking columns for remaining blocks.
#[derive(Clone)]
pub(super) struct Numeric<T> {
    /// Multifrontal tree factors for each dissected block in ordering order.
    trees: Vec<Tree<T>>,
    /// Column-by-column factors and workspace for non-dissected blocks.
    factors: ColumnFactors<T>,
    /// Flag indicating that every position holds a valid cached pivot for refactorization.
    complete: bool,
}

impl<T: Scalar> Numeric<T> {
    /// Allocates empty numeric factor storage for `ordering`.
    pub(super) fn new(ordering: &Ordering) -> Self {
        let dimension = ordering.factored().dimension();
        Self {
            trees: ordering.dissections().iter().map(Tree::new).collect(),
            factors: ColumnFactors::new(dimension),
            complete: false,
        }
    }

    /// Discards cached pivots while preserving buffer capacity so the next factorization
    /// searches fresh pivots identically to a new solver.
    pub(super) fn forget(&mut self) {
        self.complete = false;
        for tree in &mut self.trees {
            tree.forget();
        }
    }

    /// Returns whether future factorizations will select identical pivots and factor patterns
    /// as `other` (both complete with matching pivot rows and sparsity, or both empty).
    pub(super) fn pivots_alike(&self, other: &Self) -> bool {
        match (self.complete, other.complete) {
            (false, false) => true,
            (true, true) => {
                self.factors.patterns_alike(&other.factors)
                    && self
                        .trees
                        .iter()
                        .zip(&other.trees)
                        .all(|(one, another)| one.pivots_alike(another))
            }
            _ => false,
        }
    }

    /// Returns the total number of pivot columns delayed across all dissected block trees.
    pub(super) fn delays(&self) -> usize {
        self.trees.iter().map(Tree::delays).sum()
    }

    /// Returns the total stored entry count across strict lower `L` and upper `U` (including
    /// diagonal pivots).
    pub(super) fn entries(&self) -> usize {
        let dissected: usize = self.trees.iter().map(Tree::entries).sum();
        self.factors.entries() + dissected
    }

    /// Factors the matrix `values` under `ordering`, reusing prior pivots where acceptable and
    /// searching fresh pivots from the first failing position or dissected node upward.
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
                self.factors.positions.fill(NONE);
            } else {
                for &row in &self.factors.pivot_rows[first..] {
                    if row != NONE {
                        self.factors.positions[row] = NONE;
                    }
                }
            }
            self.factors.lower.truncate(first);
            self.factors.upper.truncate(first);
            self.factors.marks.fill(NONE);
            for (block, range) in ordering.block_starts().windows(2).enumerate() {
                let dissected = ordering.dissection(block).is_some();
                for column in range[0].max(first)..range[1] {
                    if dissected {
                        self.factors.pivot_rows[column] = NONE;
                        self.factors.lower.close();
                        self.factors.upper.close();
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
        for (block, range) in ordering.block_starts().windows(2).enumerate() {
            if ordering.dissection(block).is_some() {
                continue;
            }
            for column in range[0]..range[1] {
                let entries = ordering.within(column);
                let Some(pivot) = self.factors.refactor_column(entries, values, column) else {
                    return Some(column);
                };
                self.factors.pivots[column] = pivot;
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
        let entries = ordering.within(column);
        let allowed = self.factors.work.len();
        let found = self
            .factors
            .eliminate_column(entries, values, (column, column, allowed))?;
        let Some((chosen, pivot)) = found else {
            return Err(LinearError::Singular {
                columns: vec![ordering.columns()[column]],
            });
        };
        self.factors.pivot_rows[column] = chosen;
        self.factors.pivots[column] = pivot;
        Ok(())
    }

    /// Solves `A x = right_hand_side` and leaves `x` in `right_hand_side`, for the matrix of
    /// `ordering`'s pattern with `values` whose factors these are, writing to `workspace` as it
    /// goes, its dissected blocks spread over `threads` threads as [`super::SparseLu::spread`]
    /// says.
    pub(super) fn solve(
        &self,
        (ordering, values): (&Ordering, &[T]),
        right_hand_side: &mut [T],
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
        gather(by_row, right_hand_side, ordering.rows(), threads);
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
                    let value = by_row[self.factors.pivot_rows[position]];
                    by_position[position] = value;
                    self.factors.lower.subtract(position, value, by_row);
                }
                for position in positions.clone().rev() {
                    let value = by_position[position] / self.factors.pivots[position];
                    by_position[position] = value;
                    self.factors.upper.subtract(position, value, by_position);
                }
            }
            take_in_above(
                ordering.across(block),
                (values, by_position),
                (by_row, 0),
                threads,
            );
        }
        gather(
            right_hand_side,
            by_position,
            ordering.column_positions(),
            threads,
        );
    }

    /// Solves `Aᵀ x = right_hand_side` and leaves `x` in `right_hand_side`, as
    /// [`Numeric::solve`] does `A x = right_hand_side`.
    pub(super) fn solve_transpose(
        &self,
        (ordering, values): (&Ordering, &[T]),
        right_hand_side: &mut [T],
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
        gather(by_position, right_hand_side, ordering.columns(), threads);
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
                let value = take_in(above, by_position[position], (values, by_row), threads);
                let value = self
                    .factors
                    .upper
                    .subtract_dot(position, value, by_position);
                by_position[position] = value / self.factors.pivots[position];
            }
            for position in positions.rev() {
                let entry = by_position[position];
                let value = self.factors.lower.subtract_dot(position, entry, by_row);
                by_row[self.factors.pivot_rows[position]] = value;
            }
        }
        gather(right_hand_side, by_row, ordering.row_positions(), threads);
    }
}

/// Subtracts off-diagonal block contributions from `rows` above a diagonal block into
/// `by_row[row - first]`, parallelizing across rayon in [`CHUNK`] runs when `threads >= 2`.
///
/// The sequential path stays inlined so small circuits with few rows above each block incur no
/// dispatch overhead, while producing bit-identical results to parallel execution.
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

/// Splits [`take_in_above`] recursively across rayon tasks when more than [`CHUNK`] rows run
/// on two or more threads, kept out of line so the sequential path stays inlineable.
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

/// Subtracts `Σ values[slot] * solved[at]` over `entries` from `value` using deterministic
/// [`reduce::fold`] chunks, parallelizing long rows when `threads >= 2`.
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

    /// Packs the dense 3 × 3 `matrix` into slot order matching `ordering`'s row and column
    /// permutations.
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
        assert_eq!(numeric.factors.pivot_rows, [0, 1, 2]);
        assert_eq!(numeric.refactor(&ordering, &first), None);
        assert_eq!(numeric.refactor(&ordering, &second), Some(1));
        assert_eq!(
            numeric.factor(&ordering, &second, 1),
            Ok(Factorization::FellBack)
        );
        assert_eq!(numeric.factors.pivot_rows, [0, 2, 1]);
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
