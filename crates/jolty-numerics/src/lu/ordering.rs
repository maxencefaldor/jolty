//! The symbolic analysis of a pattern: the order of its rows and columns that every
//! factorization of its matrices shares.

use std::fmt;
use std::sync::Arc;

use dyn_stack::{MemBuffer, MemStack};
use faer::sparse::SymbolicSparseColMatRef;
use faer::sparse::linalg::amd;

use super::LinearError;
use crate::Pattern;

/// No index: an unmatched row or column, an unvisited vertex.
const NONE: usize = usize::MAX;

/// The analysis of a [`Pattern`] that every factorization of its matrices shares, real or
/// complex: a permutation of its rows and one of its columns, and the block triangular form
/// they put it in.
///
/// Three steps make it, as in KLU:
///
/// 1. A maximum transversal matches each column with a row holding an entry in it, keeping
///    the pattern's own diagonal entries, so that the matched entries make a diagonal free
///    of structural zeros. The equation of a voltage source, which has no entry on the
///    diagonal, is matched with one of its nodes' voltages, and that node's equation with
///    the source's current.
/// 2. The strongly connected components of the matched matrix's graph, in topological
///    order, put it in block upper triangular form. Only the diagonal blocks are factored;
///    the entries above them are read as they stand. A source fixing a node's voltage
///    against the reference becomes blocks of its own, so the many entries of a supply
///    node join no block.
/// 3. The approximate minimum degree ordering of each block's pattern plus its transpose
///    (Amestoy, Davis and Duff, "An approximate minimum degree ordering algorithm", SIAM
///    Journal on Matrix Analysis and Applications 17(4), 1996) reduces the fill of its
///    factors, its matched entries the pivots it expects.
pub struct Ordering {
    pattern: Arc<Pattern>,
    /// Whether no matching covers every column, so every matrix of the pattern is singular.
    singular: bool,
    /// The pattern's row at each position of the ordered matrix.
    rows: Vec<usize>,
    /// The pattern's column at each position of the ordered matrix.
    columns: Vec<usize>,
    /// The first position of each diagonal block, then the dimension.
    blocks: Vec<usize>,
    /// The entries of each ordered column within its diagonal block.
    within: Entries,
    /// The entries of each ordered column above its diagonal block.
    above: Entries,
}

/// Some entries of each column of the ordered matrix, by column: each its row in the
/// ordered matrix and the slot of its value in the pattern.
#[derive(Debug)]
struct Entries {
    starts: Vec<usize>,
    entries: Vec<(usize, usize)>,
}

impl Entries {
    /// No entry in any of `columns` columns.
    fn empty(columns: usize) -> Self {
        Self {
            starts: vec![0; columns + 1],
            entries: Vec::new(),
        }
    }

    /// The entries of `column`.
    fn of(&self, column: usize) -> &[(usize, usize)] {
        &self.entries[self.starts[column]..self.starts[column + 1]]
    }
}

impl Ordering {
    /// Analyzes `pattern`.
    ///
    /// A pattern no matching covers is analyzed too, as structurally singular: every
    /// factorization of its matrices fails.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::TooLarge`] when the minimum degree ordering's workspace does
    /// not fit in memory or its indices overflow.
    pub fn new(pattern: Arc<Pattern>) -> Result<Self, LinearError> {
        let dimension = pattern.dimension();
        let structure = pattern.structure();
        let (starts, rows) = (structure.col_ptr(), structure.row_idx());
        let Some(matched) = maximum_transversal(dimension, starts, rows) else {
            let identity: Vec<usize> = (0..dimension).collect();
            return Ok(Self {
                pattern,
                singular: true,
                rows: identity.clone(),
                columns: identity,
                blocks: vec![0, dimension],
                within: Entries::empty(dimension),
                above: Entries::empty(dimension),
            });
        };
        let mut column_of_row = vec![NONE; dimension];
        for (column, &row) in matched.iter().enumerate() {
            column_of_row[row] = column;
        }
        let (mut columns, blocks) = block_triangular(dimension, starts, rows, &column_of_row);
        let mut local = vec![NONE; dimension];
        for block in blocks.windows(2) {
            reduce_fill(
                &mut columns[block[0]..block[1]],
                &mut local,
                (starts, rows),
                &column_of_row,
            )?;
        }
        let ordered_rows: Vec<usize> = columns.iter().map(|&column| matched[column]).collect();
        let mut position_of_row = vec![NONE; dimension];
        for (position, &row) in ordered_rows.iter().enumerate() {
            position_of_row[row] = position;
        }
        let (mut within, mut above) = (Entries::empty(0), Entries::empty(0));
        for block in blocks.windows(2) {
            for &column in &columns[block[0]..block[1]] {
                for slot in starts[column]..starts[column + 1] {
                    let row = position_of_row[rows[slot]];
                    debug_assert!(row < block[1], "block upper triangular");
                    if row >= block[0] {
                        within.entries.push((row, slot));
                    } else {
                        above.entries.push((row, slot));
                    }
                }
                within.starts.push(within.entries.len());
                above.starts.push(above.entries.len());
            }
        }
        Ok(Self {
            pattern,
            singular: false,
            rows: ordered_rows,
            columns,
            blocks,
            within,
            above,
        })
    }

    /// The pattern analyzed.
    #[must_use]
    pub fn pattern(&self) -> &Arc<Pattern> {
        &self.pattern
    }

    /// The number of diagonal blocks of the block triangular form: one per strongly
    /// connected component of the matched matrix's graph.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.blocks.len() - 1
    }

    /// Whether every matrix of the pattern is singular, as no permutation puts a stored
    /// position on every diagonal position.
    #[must_use]
    pub fn is_singular(&self) -> bool {
        self.singular
    }

    /// The pattern's row at each position of the ordered matrix.
    pub(super) fn rows(&self) -> &[usize] {
        &self.rows
    }

    /// The pattern's column at each position of the ordered matrix.
    pub(super) fn columns(&self) -> &[usize] {
        &self.columns
    }

    /// The positions of each diagonal block, as the first of each, then the dimension.
    pub(super) fn block_starts(&self) -> &[usize] {
        &self.blocks
    }

    /// The entries of the ordered matrix's `column` within its diagonal block, each its row
    /// in the ordered matrix and the slot of its value.
    pub(super) fn within(&self, column: usize) -> &[(usize, usize)] {
        self.within.of(column)
    }

    /// The entries of the ordered matrix's `column` above its diagonal block, each its row
    /// in the ordered matrix and the slot of its value.
    pub(super) fn above(&self, column: usize) -> &[(usize, usize)] {
        self.above.of(column)
    }

    /// The number of entries above the diagonal blocks.
    pub(super) fn entries_above(&self) -> usize {
        self.above.entries.len()
    }
}

impl fmt::Debug for Ordering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ordering")
            .field("dimension", &self.pattern.dimension())
            .field("blocks", &self.blocks())
            .field("singular", &self.singular)
            .finish_non_exhaustive()
    }
}

/// The row matched with each column of the `dimension` × `dimension` pattern whose
/// columns start at `starts` in its sorted `rows`, every pair holding an entry; `None`
/// when no matching covers every column.
///
/// Each diagonal entry is matched first, then Duff's depth-first search with look-ahead
/// ("On algorithms for obtaining a maximum transversal", ACM Transactions on Mathematical
/// Software 7(3), 1981) finds an augmenting path for every column left, rematching only the
/// columns along it.
fn maximum_transversal(dimension: usize, starts: &[usize], rows: &[usize]) -> Option<Vec<usize>> {
    let column_rows = |column: usize| &rows[starts[column]..starts[column + 1]];
    let mut row_of_column = vec![NONE; dimension];
    let mut column_of_row = vec![NONE; dimension];
    for column in 0..dimension {
        if column_rows(column).binary_search(&column).is_ok() {
            row_of_column[column] = column;
            column_of_row[column] = column;
        }
    }
    // Where each column's look-ahead for an unmatched row goes on: a row it passed was
    // matched, and stays matched.
    let mut ahead = starts[..dimension].to_vec();
    // Where each column on the path goes on looking for a matched row to follow.
    let mut next = vec![0; dimension];
    // The column whose search last visited each column.
    let mut visited = vec![NONE; dimension];
    let mut path = Vec::new();
    for start in 0..dimension {
        if row_of_column[start] != NONE {
            continue;
        }
        path.clear();
        path.push(start);
        visited[start] = start;
        next[start] = starts[start];
        let mut free = None;
        while let Some(&column) = path.last() {
            let end = starts[column + 1];
            while ahead[column] < end {
                let row = rows[ahead[column]];
                ahead[column] += 1;
                if column_of_row[row] == NONE {
                    free = Some(row);
                    break;
                }
            }
            if free.is_some() {
                break;
            }
            let mut deeper = None;
            while next[column] < end {
                let other = column_of_row[rows[next[column]]];
                next[column] += 1;
                if visited[other] != start {
                    deeper = Some(other);
                    break;
                }
            }
            if let Some(other) = deeper {
                visited[other] = start;
                next[other] = starts[other];
                path.push(other);
            } else {
                path.pop();
            }
        }
        // Each column on the path takes the row that led on from it, the last the free row.
        let mut row = free?;
        for level in (0..path.len()).rev() {
            let column = path[level];
            let taken = row_of_column[column];
            row_of_column[column] = row;
            column_of_row[row] = column;
            row = taken;
        }
    }
    Some(row_of_column)
}

/// The columns of the pattern in block upper triangular order, and the first position of
/// each block then `dimension`: the strongly connected components of the graph with an
/// edge from each column to the column matched with each row holding an entry in it, by
/// Tarjan's algorithm ("Depth-first search and linear graph algorithms", SIAM Journal on
/// Computing 1(2), 1972), which finishes a component after every component it reaches,
/// so that an entry's row comes no later than its column.
fn block_triangular(
    dimension: usize,
    starts: &[usize],
    rows: &[usize],
    column_of_row: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let mut index = vec![NONE; dimension];
    let mut lowest = vec![0; dimension];
    let mut open = vec![false; dimension];
    let mut stack = Vec::with_capacity(dimension);
    // The columns being visited, each with the position of the next entry to follow.
    let mut visits: Vec<(usize, usize)> = Vec::new();
    let mut order = Vec::with_capacity(dimension);
    let mut blocks = vec![0];
    let mut count = 0;
    for root in 0..dimension {
        if index[root] != NONE {
            continue;
        }
        index[root] = count;
        lowest[root] = count;
        count += 1;
        stack.push(root);
        open[root] = true;
        visits.push((root, starts[root]));
        while let Some(&(column, entry)) = visits.last() {
            if entry < starts[column + 1] {
                if let Some(visit) = visits.last_mut() {
                    visit.1 += 1;
                }
                let successor = column_of_row[rows[entry]];
                if index[successor] == NONE {
                    index[successor] = count;
                    lowest[successor] = count;
                    count += 1;
                    stack.push(successor);
                    open[successor] = true;
                    visits.push((successor, starts[successor]));
                } else if open[successor] {
                    lowest[column] = lowest[column].min(index[successor]);
                }
                continue;
            }
            visits.pop();
            if let Some(&(parent, _)) = visits.last() {
                lowest[parent] = lowest[parent].min(lowest[column]);
            }
            if lowest[column] == index[column] {
                while let Some(member) = stack.pop() {
                    open[member] = false;
                    order.push(member);
                    if member == column {
                        break;
                    }
                }
                blocks.push(order.len());
            }
        }
    }
    (order, blocks)
}

/// Reorders the columns of one diagonal `block` by the approximate minimum degree
/// ordering of the block's pattern plus its transpose, its rows the columns matched with
/// them. `local` is a workspace of one index per column of the pattern, whose `starts` and
/// sorted `rows` are given: each column's index in its block, once its block is reached.
fn reduce_fill(
    block: &mut [usize],
    local: &mut [usize],
    (starts, rows): (&[usize], &[usize]),
    column_of_row: &[usize],
) -> Result<(), LinearError> {
    // Every ordering of two columns or fewer fills alike.
    let size = block.len();
    if size <= 2 {
        return Ok(());
    }
    for (index, &column) in block.iter().enumerate() {
        local[column] = index;
    }
    let mut block_starts = Vec::with_capacity(size + 1);
    let mut block_rows = Vec::new();
    block_starts.push(0);
    for &column in block.iter() {
        let first = block_rows.len();
        for &row in &rows[starts[column]..starts[column + 1]] {
            let matched = column_of_row[row];
            if block.get(local[matched]) == Some(&matched) {
                block_rows.push(local[matched]);
            }
        }
        block_rows[first..].sort_unstable();
        block_starts.push(block_rows.len());
    }
    let structure =
        SymbolicSparseColMatRef::new_checked(size, size, &block_starts, None, &block_rows);
    let (mut permutation, mut inverse) = (vec![0; size], vec![0; size]);
    // The workspace counts the diagonal entries, which the transpose's pattern skips, as
    // room to spare.
    let mut memory =
        MemBuffer::try_new(amd::order_scratch::<usize>(size, block_rows.len() + size))?;
    amd::order(
        &mut permutation,
        &mut inverse,
        structure,
        amd::Control::default(),
        MemStack::new(&mut memory),
    )?;
    let ordered: Vec<usize> = permutation.iter().map(|&index| block[index]).collect();
    block.copy_from_slice(&ordered);
    Ok(())
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::PatternBuilder;

    /// The pattern of a `dimension` × `dimension` matrix with an entry at each of
    /// `positions`.
    fn pattern(dimension: usize, positions: &[(usize, usize)]) -> Arc<Pattern> {
        let mut builder = PatternBuilder::new();
        for &(row, column) in positions {
            builder.entry(row, column);
        }
        Arc::new(builder.finish(dimension))
    }

    /// The row matched with each column of `pattern`.
    fn matching(pattern: &Pattern) -> Option<Vec<usize>> {
        let structure = pattern.structure();
        maximum_transversal(
            pattern.dimension(),
            structure.col_ptr(),
            structure.row_idx(),
        )
    }

    /// Whether some permutation puts an entry of `positions` on every diagonal position, by
    /// trying every permutation of the rows.
    fn perfect_matching_exists(dimension: usize, positions: &[(usize, usize)]) -> bool {
        fn extend(
            column: usize,
            used: &mut Vec<bool>,
            entry: &dyn Fn(usize, usize) -> bool,
        ) -> bool {
            if column == used.len() {
                return true;
            }
            for row in 0..used.len() {
                if !used[row] && entry(row, column) {
                    used[row] = true;
                    if extend(column + 1, used, entry) {
                        return true;
                    }
                    used[row] = false;
                }
            }
            false
        }
        let entry = |row: usize, column: usize| positions.contains(&(row, column));
        extend(0, &mut vec![false; dimension], &entry)
    }

    #[test]
    fn matches_a_voltage_sources_equation_with_its_node() {
        // A node with a conductance to the reference and a source holding it: the source's
        // current, column 1, is in the node's equation only, so the node's voltage, column
        // 0, takes the source's equation, row 1.
        let source = pattern(2, &[(0, 0), (0, 1), (1, 0)]);
        assert_eq!(matching(&source), Some(vec![1, 0]));
        // A full diagonal is kept, whatever else the pattern holds.
        let full = pattern(3, &[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, 2), (2, 2)]);
        assert_eq!(matching(&full), Some(vec![0, 1, 2]));
        // Two sources in parallel read the same node only.
        let parallel = pattern(3, &[(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]);
        assert_eq!(matching(&parallel), None);
    }

    #[test]
    fn splits_a_grounded_source_into_blocks_of_its_own() {
        // Nodes 0 and 1 joined by a conductance, node 1 held by a source whose current is
        // unknown 2: the source's current is found last, from node 1's equation, and node
        // 1's voltage first, from the source's equation.
        let held = pattern(3, &[(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]);
        let ordering = Ordering::new(held).expect("fits");
        assert_eq!(ordering.blocks(), 3);
        assert_eq!(ordering.columns(), [2, 0, 1]);
        assert_eq!(ordering.rows(), [1, 0, 2]);
    }

    proptest! {
        /// The matching covers every column exactly when some permutation does, and pairs
        /// each column with a distinct row holding an entry in it.
        #[test]
        fn finds_a_perfect_matching_when_there_is_one(
            dimension in 1usize..7,
            positions in proptest::collection::vec((0usize..7, 0usize..7), 0..30),
        ) {
            let positions: Vec<(usize, usize)> = positions
                .into_iter()
                .filter(|&(row, column)| row < dimension && column < dimension)
                .collect();
            let found = matching(&pattern(dimension, &positions));
            prop_assert_eq!(found.is_some(), perfect_matching_exists(dimension, &positions));
            if let Some(rows) = found {
                let mut taken = vec![false; dimension];
                for (column, &row) in rows.iter().enumerate() {
                    prop_assert!(positions.contains(&(row, column)));
                    prop_assert!(!taken[row]);
                    taken[row] = true;
                }
            }
        }

        /// The ordered matrix is block upper triangular with a stored diagonal, and has one
        /// block per strongly connected component of the matched matrix's graph.
        #[test]
        fn orders_the_matrix_into_its_block_triangular_form(
            dimension in 1usize..12,
            shuffle in Just((0..12).collect::<Vec<usize>>()).prop_shuffle(),
            positions in proptest::collection::vec((0usize..12, 0usize..12), 0..40),
        ) {
            // A shuffled diagonal makes every pattern matchable.
            let mut positions: Vec<(usize, usize)> = positions
                .into_iter()
                .filter(|&(row, column)| row < dimension && column < dimension)
                .collect();
            let order: Vec<usize> = shuffle.into_iter().filter(|&row| row < dimension).collect();
            positions.extend(order.iter().enumerate().map(|(column, &row)| (row, column)));
            let ordering = Ordering::new(pattern(dimension, &positions)).expect("fits");
            prop_assert!(!ordering.is_singular());
            let mut seen = (vec![false; dimension], vec![false; dimension]);
            for (&row, &column) in ordering.rows().iter().zip(ordering.columns()) {
                prop_assert!(!seen.0[row] && !seen.1[column]);
                seen.0[row] = true;
                seen.1[column] = true;
                prop_assert!(positions.contains(&(row, column)));
            }
            for block in ordering.block_starts().windows(2) {
                for position in block[0]..block[1] {
                    let column = ordering.columns()[position];
                    let count = positions
                        .iter()
                        .filter(|&&(_, other)| other == column)
                        .collect::<std::collections::BTreeSet<_>>()
                        .len();
                    let (within, above) = (ordering.within(position), ordering.above(position));
                    prop_assert_eq!(within.len() + above.len(), count);
                    prop_assert!(within.iter().all(|&(row, _)| block[0] <= row && row < block[1]));
                    prop_assert!(above.iter().all(|&(row, _)| row < block[0]));
                    prop_assert!(within.iter().any(|&(row, _)| row == position));
                }
            }
            // The components by the closure of the graph whose edge from each position runs
            // to the row of each of its entries.
            let mut reaches = vec![vec![false; dimension]; dimension];
            for (position, reach) in reaches.iter_mut().enumerate() {
                reach[position] = true;
                for &(row, _) in ordering.within(position).iter().chain(ordering.above(position)) {
                    reach[row] = true;
                }
            }
            for middle in 0..dimension {
                let onward = reaches[middle].clone();
                for reach in &mut reaches {
                    if reach[middle] {
                        for (to, &step) in reach.iter_mut().zip(&onward) {
                            *to |= step;
                        }
                    }
                }
            }
            let components = (0..dimension)
                .filter(|&position| (0..position).all(|other| !(reaches[position][other] && reaches[other][position])))
                .count();
            prop_assert_eq!(ordering.blocks(), components);
        }
    }
}
