use faer::sparse::SymbolicSparseColMat;

/// A claimed position of the system matrix, returned by [`PatternBuilder::entry`].
///
/// A device keeps its entries and writes its derivatives through them; [`Pattern::slot`] maps
/// an entry to the index of its value in the compressed matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry(usize);

/// Collects the positions of a square matrix that will ever hold a nonzero value.
///
/// Positions can be claimed before the matrix's dimension is known, as devices claim theirs
/// while internal unknowns are still being added; [`PatternBuilder::finish`] fixes it.
#[derive(Debug, Clone, Default)]
pub struct PatternBuilder {
    positions: Vec<(usize, usize)>,
}

impl PatternBuilder {
    /// Starts an empty pattern.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Claims the position at `row` and `column`. Entries claimed for the same position share
    /// one value, and their contributions add up.
    pub fn entry(&mut self, row: usize, column: usize) -> Entry {
        self.positions.push((row, column));
        Entry(self.positions.len() - 1)
    }

    /// Returns the `(row, column)` coordinates of every entry claimed so far, indexed by
    /// entry number.
    #[must_use]
    pub fn positions(&self) -> &[(usize, usize)] {
        &self.positions
    }

    /// Compresses the claimed positions into the [`Pattern`] of a `dimension` × `dimension`
    /// matrix.
    ///
    /// # Panics
    ///
    /// Panics if a claimed position lies outside the matrix.
    #[must_use]
    pub fn finish(self, dimension: usize) -> Pattern {
        #[expect(
            clippy::panic,
            reason = "every caller claims positions among the `dimension` unknowns it binds, so \
                      one outside is a bug of the caller's"
        )]
        if let Some(&(row, column)) = self
            .positions
            .iter()
            .find(|&&(row, column)| row >= dimension || column >= dimension)
        {
            panic!("entry ({row}, {column}) is outside a {dimension} × {dimension} matrix");
        }
        let mut order: Vec<usize> = (0..self.positions.len()).collect();
        order.sort_unstable_by_key(|&entry| {
            let (row, column) = self.positions[entry];
            (column, row)
        });
        let mut column_starts = vec![0; dimension + 1];
        let mut rows = Vec::with_capacity(order.len());
        let mut slots = vec![0; self.positions.len()];
        let mut previous = None;
        for entry in order {
            let position = self.positions[entry];
            if previous != Some(position) {
                rows.push(position.0);
                column_starts[position.1 + 1] += 1;
                previous = Some(position);
            }
            slots[entry] = rows.len() - 1;
        }
        for column in 0..dimension {
            column_starts[column + 1] += column_starts[column];
        }
        let mut row_starts = vec![0; dimension + 1];
        for &row in &rows {
            row_starts[row + 1] += 1;
        }
        for row in 0..dimension {
            row_starts[row + 1] += row_starts[row];
        }
        let mut filled = row_starts.clone();
        let mut by_row = vec![(0, 0); rows.len()];
        for column in 0..dimension {
            let (start, end) = (column_starts[column], column_starts[column + 1]);
            for (offset, &row) in rows[start..end].iter().enumerate() {
                by_row[filled[row]] = (start + offset, column);
                filled[row] += 1;
            }
        }
        let structure =
            SymbolicSparseColMat::new_checked(dimension, dimension, column_starts, None, rows);
        Pattern {
            structure,
            slots,
            row_starts,
            by_row,
        }
    }
}

/// Sparsity pattern of a square matrix in compressed sparse column form, mapping each
/// claimed [`Entry`] to its storage slot in the values array.
#[derive(Debug, Clone)]
pub struct Pattern {
    structure: SymbolicSparseColMat<usize>,
    slots: Vec<usize>,
    /// Starting offset of each row's entries within `by_row`.
    row_starts: Vec<usize>,
    /// Stored value slot and column index for each nonzero, grouped by row in column order.
    by_row: Vec<(usize, usize)>,
}

impl Pattern {
    /// Returns the matrix dimension (number of rows and columns).
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.structure.nrows()
    }

    /// Returns the number of distinct stored nonzero positions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.structure.row_idx().len()
    }

    /// Returns whether no nonzero positions were claimed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the compressed value-buffer slot index for `entry`.
    #[must_use]
    pub fn slot(&self, entry: Entry) -> usize {
        self.slots[entry.0]
    }

    /// Iterates over `(row, column)` coordinates of all stored values in CSC storage order.
    pub fn positions(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let starts = self.structure.col_ptr();
        let rows = self.structure.row_idx();
        (0..self.dimension()).flat_map(move |column| {
            rows[starts[column]..starts[column + 1]]
                .iter()
                .map(move |&row| (row, column))
        })
    }

    /// Adds the matrix-vector product `A x` (using `values` for stored entries) into `into`,
    /// accumulating each row in ascending column order.
    pub fn add_product(&self, values: &[f64], x: &[f64], into: &mut [f64]) {
        self.add_product_from(0, values, x, into);
    }

    /// Adds rows `first..first + into.len()` of `A x` into `into`, matching
    /// [`Pattern::add_product`] bit for bit so row slices can execute in parallel.
    pub fn add_product_from(&self, first: usize, values: &[f64], x: &[f64], into: &mut [f64]) {
        for (offset, value) in into.iter_mut().enumerate() {
            *value = self
                .row(first + offset)
                .iter()
                .fold(*value, |total, &(slot, column)| {
                    total + values[slot] * x[column]
                });
        }
    }

    /// Returns `(slot, column)` entries for `row` in ascending column order, matching
    /// [`Pattern::positions`].
    #[must_use]
    pub fn row(&self, row: usize) -> &[(usize, usize)] {
        let (Some(&start), Some(&end)) = (self.row_starts.get(row), self.row_starts.get(row + 1))
        else {
            return &[];
        };
        &self.by_row[start..end]
    }

    pub(crate) fn structure(&self) -> &SymbolicSparseColMat<usize> {
        &self.structure
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_repeated_positions_and_orders_by_column() {
        let mut builder = PatternBuilder::new();
        let a = builder.entry(2, 0);
        let b = builder.entry(0, 0);
        let c = builder.entry(2, 0);
        let d = builder.entry(1, 2);
        let pattern = builder.finish(3);
        assert_eq!(pattern.len(), 3);
        assert_eq!(
            pattern.positions().collect::<Vec<_>>(),
            [(0, 0), (2, 0), (1, 2)]
        );
        assert_eq!(pattern.slot(a), pattern.slot(c));
        assert_eq!(
            [pattern.slot(b), pattern.slot(a), pattern.slot(d)],
            [0, 1, 2]
        );
        // Each row's entries, by stored position and column, in the order of the columns.
        assert_eq!(
            [pattern.row(0), pattern.row(1), pattern.row(2)],
            [&[(0, 0)][..], &[(2, 2)], &[(1, 0)]]
        );
    }

    #[test]
    fn adds_its_product_with_a_vector() {
        let mut builder = PatternBuilder::new();
        builder.entry(0, 0);
        builder.entry(1, 0);
        builder.entry(0, 1);
        let pattern = builder.finish(2);
        // [[1, 3], [2, 0]] · [1, 10] = [31, 2], added to [1, 1].
        let mut into = [1.0, 1.0];
        pattern.add_product(&[1.0, 2.0, 3.0], &[1.0, 10.0], &mut into);
        assert!((into[0] - 32.0).abs() < 1e-15 && (into[1] - 3.0).abs() < 1e-15);
        // The second row alone, as a thread that takes it adds it.
        let mut second = [1.0];
        pattern.add_product_from(1, &[1.0, 2.0, 3.0], &[1.0, 10.0], &mut second);
        assert_eq!(second[0].to_bits(), into[1].to_bits());
    }

    proptest::proptest! {
        /// Verifies that row-wise accumulation matches column-wise storage-order accumulation
        /// bit for bit from any starting row.
        #[test]
        fn a_product_by_rows_is_the_product_by_columns(
            positions in proptest::collection::vec((0..30usize, 0..30usize), 1..200),
            first in 0..30usize,
        ) {
            let mut builder = PatternBuilder::new();
            for &(row, column) in &positions {
                builder.entry(row, column);
            }
            let pattern = builder.finish(30);
            let values: Vec<f64> = (0..pattern.len())
                .map(|slot| (f64::from(u32::try_from(slot).unwrap()) * 0.7).sin())
                .collect();
            let x: Vec<f64> = (0..30).map(|column| (f64::from(column) * 1.3).cos()).collect();
            let start: Vec<f64> = (0..30).map(|row| f64::from(row) * 0.1).collect();
            let mut by_columns = start.clone();
            for ((row, column), value) in pattern.positions().zip(&values) {
                by_columns[row] += value * x[column];
            }
            let mut by_rows = start.clone();
            pattern.add_product(&values, &x, &mut by_rows);
            let mut later = start[first..].to_vec();
            pattern.add_product_from(first, &values, &x, &mut later);
            let bits = |values: &[f64]| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
            proptest::prop_assert_eq!(bits(&by_rows), bits(&by_columns));
            proptest::prop_assert_eq!(bits(&later), bits(&by_columns[first..]));
        }
    }

    #[test]
    #[should_panic(expected = "outside a 2 × 2 matrix")]
    fn rejects_positions_outside_the_matrix() {
        let mut builder = PatternBuilder::new();
        builder.entry(2, 0);
        let _ = builder.finish(2);
    }
}
