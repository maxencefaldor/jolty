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

use super::{LinearError, Ordering, Scalar};

/// No index: a row not yet a pivot, a row not yet visited.
const NONE: usize = usize::MAX;

/// The least fraction of the largest candidate in its column that a pivot is: a
/// factorization takes the ordering's diagonal entry as the pivot when it reaches this
/// fraction, and else the largest, and a refactorization keeps each pivot while it reaches
/// it. It bounds every entry of `L` by its inverse, ten, against a thousand for the
/// thousandth that KLU and SPICE take, which keeps the rounding of the solutions close to
/// that of strict partial pivoting: over 20,000 of the tests' random circuits, the largest
/// backward error of a solution is 4.5e-15 with a tenth, against 7.7e-9 with a thousandth.
/// On the matrices of the example circuits, meshes and inverter chains measured, the
/// factors fill alike with either, but for one entry of one. The price is that a
/// refactorization keeps its pivots less often: in the transients of the examples' buck and
/// boost converters, one or two factorizations in a hundred search afresh, where with a
/// thousandth almost none does.
const PIVOT_TOLERANCE: f64 = 0.1;

/// A triangular factor stored by columns.
struct Triangle<T> {
    /// Where each column's entries start, then their number.
    starts: Vec<usize>,
    /// Each entry's index: a row of the ordered matrix in `L`, a position in `U`.
    indices: Vec<usize>,
    /// Each entry's value.
    values: Vec<T>,
}

impl<T: Scalar> Triangle<T> {
    fn new(dimension: usize) -> Self {
        let mut starts = Vec::with_capacity(dimension + 1);
        starts.push(0);
        Self {
            starts,
            indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// The indices and values of `column`'s entries.
    fn column(&self, column: usize) -> (&[usize], &[T]) {
        let range = self.starts[column]..self.starts[column + 1];
        (&self.indices[range.clone()], &self.values[range])
    }

    /// Keeps the first `columns` columns.
    fn truncate(&mut self, columns: usize) {
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
/// each block by its pivots, and the workspace that finds them.
pub(super) struct Numeric<T> {
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
    pub(super) fn new(dimension: usize) -> Self {
        Self {
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
    pub(super) const fn forget(&mut self) {
        self.complete = false;
    }

    /// The number of entries of `L` below its diagonal and of `U` on and above it.
    pub(super) fn entries(&self) -> usize {
        self.lower.indices.len() + self.upper.indices.len() + self.pivots.len()
    }

    /// Factors the matrix of `ordering`'s pattern whose values are `values`, keeping the
    /// last factorization's pivots while each passes, and searching for the pivots from the
    /// first that does not. Returns how it went.
    pub(super) fn factor(
        &mut self,
        ordering: &Ordering,
        values: &[T],
    ) -> Result<Factorization, LinearError> {
        let (first, went) = if self.complete {
            match self.refactor(ordering, values) {
                None => return Ok(Factorization::Refactored),
                Some(first) => (first, Factorization::FellBack),
            }
        } else {
            (0, Factorization::Searched)
        };
        if self.complete {
            for &row in &self.pivot_rows[first..] {
                self.positions[row] = NONE;
            }
        } else {
            self.positions.fill(NONE);
        }
        self.complete = false;
        self.lower.truncate(first);
        self.upper.truncate(first);
        self.marks.fill(NONE);
        for column in first..self.pivots.len() {
            self.eliminate(ordering, values, column)?;
        }
        self.complete = true;
        Ok(went)
    }

    /// Refactors with the pivots and patterns of the last factorization. Returns `None`
    /// when every pivot reaches its tolerance, and else the first position whose pivot does
    /// not, the factors before it found.
    fn refactor(&mut self, ordering: &Ordering, values: &[T]) -> Option<usize> {
        for column in 0..self.pivots.len() {
            for &(row, slot) in ordering.within(column) {
                self.work[row] = values[slot];
            }
            for entry in self.upper.starts[column]..self.upper.starts[column + 1] {
                let position = self.upper.indices[entry];
                let row = self.pivot_rows[position];
                let multiplier = self.work[row];
                self.work[row] = T::ZERO;
                self.upper.values[entry] = multiplier;
                let (indices, factors) = self.lower.column(position);
                for (&index, &factor) in indices.iter().zip(factors) {
                    self.work[index] -= factor * multiplier;
                }
            }
            let row = self.pivot_rows[column];
            let pivot = self.work[row];
            self.work[row] = T::ZERO;
            let below = self.lower.starts[column]..self.lower.starts[column + 1];
            let largest = self.lower.indices[below.clone()]
                .iter()
                .map(|&index| self.work[index].magnitude())
                .fold(0.0, f64::max);
            let magnitude = pivot.magnitude();
            let passes = magnitude > 0.0 && magnitude >= PIVOT_TOLERANCE * largest;
            for entry in below {
                let index = self.lower.indices[entry];
                self.lower.values[entry] = self.work[index] / pivot;
                self.work[index] = T::ZERO;
            }
            if !passes {
                return Some(column);
            }
            self.pivots[column] = pivot;
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
            while let Some(&(row, entry)) = self.visits.last() {
                let position = self.positions[row];
                if position != NONE && entry < self.lower.starts[position + 1] {
                    if let Some(visit) = self.visits.last_mut() {
                        visit.1 += 1;
                    }
                    let next = self.lower.indices[entry];
                    if self.marks[next] != column {
                        self.marks[next] = column;
                        self.visits.push((next, self.first_entry(next)));
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
    /// with `values` whose factors these are, with the workspaces `by_row` and
    /// `by_position`.
    pub(super) fn solve(
        &self,
        ordering: &Ordering,
        values: &[T],
        rhs: &mut [T],
        (by_row, by_position): (&mut [T], &mut [T]),
    ) {
        for (value, &row) in by_row.iter_mut().zip(ordering.rows()) {
            *value = rhs[row];
        }
        for block in ordering.block_starts().windows(2).rev() {
            let positions = block[0]..block[1];
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
            for position in positions {
                let value = by_position[position];
                for &(row, slot) in ordering.above(position) {
                    by_row[row] -= values[slot] * value;
                }
            }
        }
        for (&column, &value) in ordering.columns().iter().zip(by_position.iter()) {
            rhs[column] = value;
        }
    }

    /// Solves `Aᵀ x = rhs` and leaves `x` in `rhs`, as [`Numeric::solve`] does `A x = rhs`.
    pub(super) fn solve_transpose(
        &self,
        ordering: &Ordering,
        values: &[T],
        rhs: &mut [T],
        (by_row, by_position): (&mut [T], &mut [T]),
    ) {
        for (value, &column) in by_position.iter_mut().zip(ordering.columns()) {
            *value = rhs[column];
        }
        for block in ordering.block_starts().windows(2) {
            let positions = block[0]..block[1];
            for position in positions.clone() {
                let mut value = by_position[position];
                for &(row, slot) in ordering.above(position) {
                    value -= values[slot] * by_row[row];
                }
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
        for (&row, &value) in ordering.rows().iter().zip(by_row.iter()) {
            rhs[row] = value;
        }
    }
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
        let mut numeric = Numeric::new(3);
        assert_eq!(
            numeric.factor(&ordering, &first),
            Ok(Factorization::Searched)
        );
        assert_eq!(numeric.pivot_rows, [0, 1, 2]);
        assert_eq!(numeric.refactor(&ordering, &first), None);
        assert_eq!(numeric.refactor(&ordering, &second), Some(1));
        assert_eq!(
            numeric.factor(&ordering, &second),
            Ok(Factorization::FellBack)
        );
        assert_eq!(numeric.pivot_rows, [0, 2, 1]);
        // The factors solve the second matrix's system: its residual is a rounding.
        let wanted = [1.0, -2.0, 3.0];
        let mut x = wanted;
        let mut workspace = ([0.0; 3], [0.0; 3]);
        numeric.solve(
            &ordering,
            &second,
            &mut x,
            (&mut workspace.0, &mut workspace.1),
        );
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
