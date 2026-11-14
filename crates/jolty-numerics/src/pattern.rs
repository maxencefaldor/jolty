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
        let structure =
            SymbolicSparseColMat::new_checked(dimension, dimension, column_starts, None, rows);
        Pattern { structure, slots }
    }
}

/// The sparsity pattern of a square matrix in compressed sparse column form, and where each
/// claimed [`Entry`] lands among its values.
#[derive(Debug, Clone)]
pub struct Pattern {
    structure: SymbolicSparseColMat<usize>,
    slots: Vec<usize>,
}

impl Pattern {
    /// The number of rows and columns.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.structure.nrows()
    }

    /// The number of stored values: the distinct positions claimed.
    #[must_use]
    pub fn len(&self) -> usize {
        self.structure.row_idx().len()
    }

    /// Whether no position was claimed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The index of `entry`'s value among the matrix values.
    #[must_use]
    pub fn slot(&self, entry: Entry) -> usize {
        self.slots[entry.0]
    }

    /// The row and column of each stored value, in storage order.
    pub fn positions(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let starts = self.structure.col_ptr();
        let rows = self.structure.row_idx();
        (0..self.dimension()).flat_map(move |column| {
            rows[starts[column]..starts[column + 1]]
                .iter()
                .map(move |&row| (row, column))
        })
    }

    /// Adds the product of the matrix whose stored values are `values` and the vector `x`
    /// to `into`.
    pub fn add_product(&self, values: &[f64], x: &[f64], into: &mut [f64]) {
        for ((row, column), value) in self.positions().zip(values) {
            into[row] += value * x[column];
        }
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
    }

    #[test]
    #[should_panic(expected = "outside a 2 × 2 matrix")]
    fn rejects_positions_outside_the_matrix() {
        let mut builder = PatternBuilder::new();
        builder.entry(2, 0);
        let _ = builder.finish(2);
    }
}
