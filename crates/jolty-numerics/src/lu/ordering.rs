//! Symbolic analysis of a pattern: the row and column ordering shared by every factorization
//! of its matrices.

use std::fmt;
use std::sync::Arc;

use dyn_stack::{MemBuffer, MemStack};
use faer::sparse::SymbolicSparseColMatRef;
use faer::sparse::linalg::amd;

use super::condensation::Condensation;
use super::dissection::{self, Dissection, Graph};
use super::{LinearError, NONE};
use crate::Pattern;

/// Symbolic analysis of a [`Pattern`] shared by every real or complex factorization of its
/// matrices: row and column permutations together with the resulting block upper triangular
/// form.
///
/// Analysis proceeds in three steps, following KLU:
///
/// 1. A maximum transversal matches each column with a row holding a stored entry while
///    preserving existing diagonal entries. The matched diagonal is free of structural zeros.
///    A voltage source equation lacks a diagonal entry, so it is matched with one of its
///    terminal voltages and that node equation is matched with the source current.
/// 2. Strongly connected components of the matched graph in topological order yield a block
///    upper triangular form. Only diagonal blocks require factorization; off-diagonal entries
///    above them remain unchanged. A grounded voltage source forms singleton blocks, keeping
///    supply-node entries out of larger blocks.
/// 3. Approximate minimum degree ordering on each block pattern plus its transpose (Amestoy,
///    Davis and Duff, "An approximate minimum degree ordering algorithm", SIAM Journal on
///    Matrix Analysis and Applications 17(4), 1996) reduces factor fill around the matched
///    pivots.
///
/// An ordering [dissected](Ordering::dissected) for a given interior limit partitions larger
/// diagonal blocks by nested dissection instead. Each resulting interior uses minimum degree
/// ordering, and separating fronts factor in parallel according to the configured
/// [`Dissecting`] policy.
///
/// An ordering of a [`Condensation`] analyzes only the kept unknowns. Solvers built from the
/// ordering condense each input matrix before factorization and recover eliminated unknowns
/// after each solve.
pub struct Ordering {
    /// Analyzed pattern (the kept-unknown pattern when condensed).
    pattern: Arc<Pattern>,
    /// Condensation mapping full matrices onto the analyzed pattern when unknowns are eliminated.
    condensation: Option<Arc<Condensation>>,
    /// Sorted subset of columns spanning fewer rows than columns when structurally singular, or
    /// empty otherwise.
    undetermined: Vec<usize>,
    /// Original pattern row index at each ordered matrix position.
    rows: Vec<usize>,
    /// Original pattern column index at each ordered matrix position.
    columns: Vec<usize>,
    /// Ordered matrix position of each pattern row and column.
    row_positions: Vec<usize>,
    column_positions: Vec<usize>,
    /// Starting position of each diagonal block, terminated by the matrix dimension.
    blocks: Vec<usize>,
    /// Ordered-column entries lying within each diagonal block.
    within: Entries,
    /// Ordered-column entries lying above each diagonal block.
    above: Entries,
    /// Off-diagonal entries above each diagonal block, indexed by row.
    across: Across,
    /// Policy controlling which diagonal blocks use nested dissection.
    dissecting: Option<Dissecting>,
    /// Nested dissections of the dissected blocks, in block order.
    dissections: Vec<Dissection>,
    /// Index into `dissections` for each block, or `NONE` when not dissected.
    dissected: Vec<usize>,
}

/// Policy selecting which diagonal blocks an [`Ordering`] partitions by nested dissection
/// instead of minimum degree ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Dissecting {
    /// Every diagonal block larger than the given interior size limit.
    Every(usize),
    /// Every diagonal block larger than the interior limit for which nested dissection needs
    /// fewer elimination multiply-adds than minimum degree ordering.
    ///
    /// Both counts come from the symbolic factor of the symmetrized block pattern, counting
    /// one multiply-add for each product between column `L` entries and row `U` entries. On a
    /// square grid with interior limit 157, nested dissection uses 1.06 times the minimum
    /// degree count at 60 nodes per side and 0.96 times at 70. On a path of 5,000 nodes, where
    /// minimum degree incurs zero fill, nested dissection uses 3.9 times as many multiply-adds.
    Cheaper(usize),
}

impl Dissecting {
    /// Returns the maximum number of positions per interior, clamped to at least one.
    #[must_use]
    pub fn interiors(self) -> usize {
        match self {
            Self::Every(most) | Self::Cheaper(most) => most.max(1),
        }
    }
}

/// Sparse column entries stored as `(row, value_slot)` pairs.
#[derive(Debug)]
pub(super) struct Entries {
    /// Starting offset of each column in `entries`, terminated by the total count.
    pub(super) starts: Vec<usize>,
    /// Packed `(row, value_slot)` pairs across all columns.
    pub(super) entries: Vec<(usize, usize)>,
}

impl Entries {
    /// Constructs an empty entry list with `columns` columns.
    pub(super) fn empty(columns: usize) -> Self {
        Self {
            starts: vec![0; columns + 1],
            entries: Vec::new(),
        }
    }

    /// Returns the stored entries of `column`.
    pub(super) fn of(&self, column: usize) -> &[(usize, usize)] {
        &self.entries[self.starts[column]..self.starts[column + 1]]
    }
}

/// Off-diagonal entries above each diagonal block, grouped by ascending row and column position.
#[derive(Debug)]
pub(super) struct Across {
    /// Starting offset of each block in `rows`, terminated by the total row count.
    blocks: Vec<usize>,
    /// Sorted row indices with entries above each diagonal block.
    rows: Vec<usize>,
    /// Starting offset of each row in `entries`, terminated by the total entry count.
    starts: Vec<usize>,
    /// Packed `(column_position, value_slot)` pairs across all rows.
    entries: Vec<(usize, usize)>,
}

impl Across {
    /// Constructs an empty off-diagonal row index for `blocks` diagonal blocks.
    fn empty(blocks: usize) -> Self {
        Self {
            blocks: vec![0; blocks + 1],
            rows: Vec::new(),
            starts: vec![0],
            entries: Vec::new(),
        }
    }

    /// Transposes column-major `above` entries into row-major form for each diagonal block
    /// defined by `blocks`.
    fn new(blocks: &[usize], above: &Entries) -> Self {
        let mut across = Self::empty(0);
        across.entries.reserve_exact(above.entries.len());
        let mut found = Vec::new();
        for block in blocks.windows(2) {
            found.clear();
            for position in block[0]..block[1] {
                found.extend(
                    above
                        .of(position)
                        .iter()
                        .map(|&(row, slot)| (row, position, slot)),
                );
            }
            // Stable sort keeps each row's entries in ascending column position.
            found.sort_by_key(|&(row, _, _)| row);
            for entries in found.chunk_by(|one, other| one.0 == other.0) {
                across.rows.push(entries[0].0);
                across
                    .entries
                    .extend(entries.iter().map(|&(_, position, slot)| (position, slot)));
                across.starts.push(across.entries.len());
            }
            across.blocks.push(across.rows.len());
        }
        across
    }

    /// Returns the rows above `block` and their entries in the columns of `block`.
    #[inline]
    pub(super) fn of(&self, block: usize) -> RowsAbove<'_> {
        let (first, end) = (self.blocks[block], self.blocks[block + 1]);
        RowsAbove {
            rows: &self.rows[first..end],
            starts: &self.starts[first..=end],
            entries: &self.entries,
        }
    }
}

/// Borrowed view of rows above a diagonal block and their entries in ascending position order.
#[derive(Debug, Clone, Copy)]
pub(super) struct RowsAbove<'a> {
    /// Sorted row indices above the block.
    rows: &'a [usize],
    /// Starting offset of each row in `entries`, terminated by the end offset of the final row.
    starts: &'a [usize],
    /// Packed `(column_position, value_slot)` pairs.
    entries: &'a [(usize, usize)],
}

impl<'a> RowsAbove<'a> {
    /// Returns the number of rows in the view.
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.rows.len()
    }

    /// Returns the row index at `index` in ascending order.
    #[inline]
    pub(super) fn row(&self, index: usize) -> usize {
        self.rows[index]
    }

    /// Returns an iterator over each row index and its slice of entries.
    #[inline]
    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, &'a [(usize, usize)])> + use<'a> {
        let Self {
            rows,
            starts,
            entries,
        } = *self;
        rows.iter()
            .zip(starts.windows(2))
            .map(move |(&row, bounds)| (row, &entries[bounds[0]..bounds[1]]))
    }

    /// Splits the view into the rows before `index` and the rows from `index` onward.
    pub(super) fn split_at(&self, index: usize) -> (Self, Self) {
        let (before, after) = self.rows.split_at(index);
        (
            Self {
                rows: before,
                starts: &self.starts[..=index],
                entries: self.entries,
            },
            Self {
                rows: after,
                starts: &self.starts[index..],
                entries: self.entries,
            },
        )
    }
}

impl Ordering {
    /// Analyzes `pattern` without nested dissection.
    ///
    /// A structurally singular pattern with no full matching is still analyzed; every
    /// subsequent factorization fails.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::TooLarge`] when the minimum degree ordering workspace does not
    /// fit in memory or its indices overflow.
    pub fn new(pattern: Arc<Pattern>) -> Result<Self, LinearError> {
        Self::analyze(pattern, None)
    }

    /// Analyzes `pattern` as [`Ordering::new`] does and orders selected diagonal blocks by
    /// nested dissection according to `dissecting`.
    ///
    /// Each dissected block splits into interiors of at most [`Dissecting::interiors`]
    /// positions unless no further split exists. Interiors use minimum degree ordering,
    /// followed by separators ordered after the subtrees they separate. Factorization eliminates
    /// interiors in parallel with local pivoting and then eliminates each separator once its
    /// children complete. Operation order depends only on the pattern and `dissecting`, so the
    /// resulting factor bits are independent of thread count.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::TooLarge`] when the minimum degree ordering workspace does not
    /// fit in memory or its indices overflow.
    pub fn dissected(pattern: Arc<Pattern>, dissecting: Dissecting) -> Result<Self, LinearError> {
        Self::analyze(pattern, Some(dissecting))
    }

    /// Analyzes the kept-unknown pattern of `condensation`, optionally dissecting blocks
    /// according to `dissecting`.
    ///
    /// Solvers built from the returned ordering accept matrices over the full pattern of
    /// `condensation` and condense each matrix before factorization.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::TooLarge`] when the minimum degree ordering workspace does not
    /// fit in memory or its indices overflow.
    pub fn condensed(
        condensation: Arc<Condensation>,
        dissecting: Option<Dissecting>,
    ) -> Result<Self, LinearError> {
        if condensation.eliminated() == 0 {
            return Self::analyze(Arc::clone(condensation.full()), dissecting);
        }
        let mut ordering = Self::analyze(Arc::clone(condensation.kept()), dissecting)?;
        for column in &mut ordering.undetermined {
            *column = condensation.unknown(*column);
        }
        ordering.condensation = Some(condensation);
        Ok(ordering)
    }

    /// Constructs a structurally singular ordering recording `undetermined` columns.
    fn singular(
        pattern: Arc<Pattern>,
        undetermined: Vec<usize>,
        dissecting: Option<Dissecting>,
    ) -> Self {
        let dimension = pattern.dimension();
        let identity: Vec<usize> = (0..dimension).collect();
        Self {
            pattern,
            condensation: None,
            undetermined,
            rows: identity.clone(),
            columns: identity.clone(),
            row_positions: identity.clone(),
            column_positions: identity,
            blocks: vec![0, dimension],
            within: Entries::empty(dimension),
            above: Entries::empty(dimension),
            across: Across::empty(1),
            dissecting,
            dissections: Vec::new(),
            dissected: vec![NONE],
        }
    }

    /// Analyzes `pattern`, applying nested dissection according to `dissecting` when provided.
    fn analyze(pattern: Arc<Pattern>, dissecting: Option<Dissecting>) -> Result<Self, LinearError> {
        let dimension = pattern.dimension();
        let structure = pattern.structure();
        let (starts, rows) = (structure.col_ptr(), structure.row_idx());
        let matched = match maximum_transversal(dimension, starts, rows) {
            Ok(matched) => matched,
            Err(undetermined) => return Ok(Self::singular(pattern, undetermined, dissecting)),
        };
        let column_of_row = invert(&matched);
        let (mut columns, blocks) = block_triangular(dimension, starts, rows, &column_of_row);
        let mut local = vec![NONE; dimension];
        // The tree of each dissected block, by block.
        let mut trees = Vec::new();
        for (index, block) in blocks.windows(2).enumerate() {
            let range = &mut columns[block[0]..block[1]];
            let found = order_block(
                range,
                &mut local,
                ((starts, rows), &column_of_row),
                dissecting,
            )?;
            if let Some(found) = found {
                trees.push((index, found));
            }
        }
        let ordered_rows: Vec<usize> = columns.iter().map(|&column| matched[column]).collect();
        let position_of_row = invert(&ordered_rows);
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
        let across = Across::new(&blocks, &above);
        let mut dissected = vec![NONE; blocks.len() - 1];
        let dissections = trees
            .into_iter()
            .enumerate()
            .map(|(order, (block, found))| {
                dissected[block] = order;
                Dissection::new((blocks[block], blocks[block + 1]), &found, |column| {
                    within.of(column)
                })
            })
            .collect();
        let column_positions = invert(&columns);
        Ok(Self {
            pattern,
            condensation: None,
            undetermined: Vec::new(),
            rows: ordered_rows,
            columns,
            row_positions: position_of_row,
            column_positions,
            blocks,
            within,
            above,
            across,
            dissecting,
            dissections,
            dissected,
        })
    }

    /// Returns the full matrix pattern accepted by solvers built from the ordering.
    #[must_use]
    pub fn pattern(&self) -> &Arc<Pattern> {
        self.condensation
            .as_ref()
            .map_or(&self.pattern, |condensation| condensation.full())
    }

    /// Returns the pattern actually factored: the kept-unknown pattern when condensed, or
    /// [`Ordering::pattern`] otherwise.
    #[must_use]
    pub fn factored(&self) -> &Arc<Pattern> {
        &self.pattern
    }

    /// Returns the active condensation when at least one unknown is eliminated.
    pub(super) fn condensation(&self) -> Option<&Condensation> {
        self.condensation.as_deref()
    }

    /// Returns the number of diagonal blocks in the block upper triangular form.
    #[must_use]
    pub fn blocks(&self) -> usize {
        self.blocks.len() - 1
    }

    /// Returns `true` when every matrix of the pattern is structurally singular.
    #[must_use]
    pub fn is_singular(&self) -> bool {
        !self.undetermined.is_empty()
    }

    /// Returns a sorted subset of columns of [`Ordering::pattern`] whose entries lie in fewer
    /// rows than columns when structurally singular, or an empty slice otherwise.
    #[must_use]
    pub fn undetermined(&self) -> &[usize] {
        &self.undetermined
    }

    /// Returns the pattern row index at each ordered matrix position.
    pub(super) fn rows(&self) -> &[usize] {
        &self.rows
    }

    /// Returns the pattern column index at each ordered matrix position.
    pub(super) fn columns(&self) -> &[usize] {
        &self.columns
    }

    /// Returns the ordered matrix position of each pattern row.
    pub(super) fn row_positions(&self) -> &[usize] {
        &self.row_positions
    }

    /// Returns the ordered matrix position of each pattern column.
    pub(super) fn column_positions(&self) -> &[usize] {
        &self.column_positions
    }

    /// Returns the starting position of each diagonal block, followed by the matrix dimension.
    pub(super) fn block_starts(&self) -> &[usize] {
        &self.blocks
    }

    /// Returns the `(ordered_row, value_slot)` entries of ordered `column` inside its diagonal
    /// block.
    pub(super) fn within(&self, column: usize) -> &[(usize, usize)] {
        self.within.of(column)
    }

    /// Returns the `(ordered_row, value_slot)` entries of ordered `column` above its diagonal
    /// block.
    pub(super) fn above(&self, column: usize) -> &[(usize, usize)] {
        self.above.of(column)
    }

    /// Returns the rows above diagonal `block` together with their `(ordered_column,
    /// value_slot)` entries.
    #[inline]
    pub(super) fn across(&self, block: usize) -> RowsAbove<'_> {
        self.across.of(block)
    }

    /// Returns the total number of stored entries above the diagonal blocks.
    pub(super) fn entries_above(&self) -> usize {
        self.above.entries.len()
    }

    /// Returns the dimension of the largest diagonal block.
    #[must_use]
    pub fn largest_block(&self) -> usize {
        self.blocks
            .windows(2)
            .map(|block| block[1] - block[0])
            .max()
            .unwrap_or(0)
    }

    /// Returns `true` when at least one diagonal block uses nested dissection.
    #[must_use]
    pub fn is_dissected(&self) -> bool {
        !self.dissections.is_empty()
    }

    /// Returns the configured [`Dissecting::interiors`] limit if the ordering was
    /// [dissected](Ordering::dissected).
    #[must_use]
    pub fn interiors(&self) -> Option<usize> {
        self.dissecting.map(Dissecting::interiors)
    }

    /// Returns the nested dissections of all dissected blocks, in block order.
    pub(super) fn dissections(&self) -> &[Dissection] {
        &self.dissections
    }

    /// Returns the nested dissection for `block`, if `block` is dissected.
    pub(super) fn dissection(&self, block: usize) -> Option<&Dissection> {
        self.dissections.get(self.dissected[block])
    }

    /// Returns the dissection table index for `block`, or `NONE` when `block` is not dissected.
    pub(super) fn dissected_index(&self, block: usize) -> usize {
        self.dissected[block]
    }
}

impl fmt::Debug for Ordering {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ordering")
            .field("dimension", &self.pattern.dimension())
            .field("blocks", &self.blocks())
            .field("undetermined", &self.undetermined)
            .field("dissecting", &self.dissecting)
            .field("dissected", &self.dissections.len())
            .finish_non_exhaustive()
    }
}

/// Finds a row-to-column matching covering every column of the `dimension` × `dimension`
/// pattern given by `starts` and sorted `rows`, or returns a sorted subset of columns whose
/// entries span fewer rows than columns.
///
/// Existing diagonal entries are matched first. Duff's depth-first search with look-ahead
/// ("On algorithms for obtaining a maximum transversal", ACM Transactions on Mathematical
/// Software 7(3), 1981) then finds an augmenting path for each remaining column, rematching
/// only columns along the path. A failed search visits a Hall witness of columns reachable by
/// alternating paths whose rows all match within that set, yielding one more column than rows.
fn maximum_transversal(
    dimension: usize,
    starts: &[usize],
    rows: &[usize],
) -> Result<Vec<usize>, Vec<usize>> {
    let column_rows = |column: usize| &rows[starts[column]..starts[column + 1]];
    let mut row_of_column = vec![NONE; dimension];
    let mut column_of_row = vec![NONE; dimension];
    for column in 0..dimension {
        if column_rows(column).binary_search(&column).is_ok() {
            row_of_column[column] = column;
            column_of_row[column] = column;
        }
    }
    // Look-ahead cursor for an unmatched row in each column: passed rows stay matched.
    let mut ahead = starts[..dimension].to_vec();
    // Traversal cursor for matched rows along the current augmenting path.
    let mut next = vec![0; dimension];
    // Search root that most recently visited each column.
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
        let Some(mut row) = free else {
            return Err((0..dimension)
                .filter(|&column| visited[column] == start)
                .collect());
        };
        // Each column on the path takes the row that led on from it, the last the free row.
        for level in (0..path.len()).rev() {
            let column = path[level];
            let taken = row_of_column[column];
            row_of_column[column] = row;
            column_of_row[row] = column;
            row = taken;
        }
    }
    Ok(row_of_column)
}

/// Computes the block upper triangular column order and block boundary offsets for the
/// matched pattern.
///
/// Tarjan's strongly connected components algorithm ("Depth-first search and linear graph
/// algorithms", SIAM Journal on Computing 1(2), 1972) walks edges from each column to the
/// column matched with each row entry in that column. Each component finishes after all
/// reachable components, ensuring that every stored entry has a row no later than its column.
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
        while let Some(visit) = visits.last_mut() {
            let (column, entry) = *visit;
            if entry < starts[column + 1] {
                visit.1 += 1;
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

/// Returns the inverse permutation of `order` over `0..order.len()`.
fn invert(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![NONE; order.len()];
    for (position, &item) in order.iter().enumerate() {
        inverse[item] = position;
    }
    inverse
}

/// Writes each column's index within `block` into `local`.
fn index_block(block: &[usize], local: &mut [usize]) {
    for (index, &column) in block.iter().enumerate() {
        local[column] = index;
    }
}

/// Returns local indices within `block` of the matched columns referenced by `column` in
/// `(starts, rows)`.
fn block_neighbors(
    block: &[usize],
    local: &[usize],
    (starts, rows): (&[usize], &[usize]),
    column_of_row: &[usize],
    column: usize,
) -> impl Iterator<Item = usize> {
    let entries = &rows[starts[column]..starts[column + 1]];
    entries.iter().filter_map(move |&row| {
        let matched = column_of_row[row];
        let index = local[matched];
        (block.get(index) == Some(&matched)).then_some(index)
    })
}

/// Orders one diagonal `block` by nested dissection into interiors of at most `most` columns,
/// reorders each interior by minimum degree, and returns the postorder tree nodes.
///
/// Leaves `block` unchanged and returns `None` when the block cannot split. Uses `local` as
/// scratch space following [`reduce_fill`].
fn dissect(
    block: &mut [usize],
    local: &mut [usize],
    (starts, rows): (&[usize], &[usize]),
    column_of_row: &[usize],
    most: usize,
) -> Result<Option<Vec<dissection::Found>>, LinearError> {
    index_block(block, local);
    let lists: Vec<Vec<usize>> = block
        .iter()
        .map(|&column| {
            block_neighbors(block, local, (starts, rows), column_of_row, column).collect()
        })
        .collect();
    let found = dissection::dissect(&Graph::symmetric(&lists), most);
    if found.len() < 2 {
        return Ok(None);
    }
    let ordered: Vec<usize> = found
        .iter()
        .flat_map(|node| node.vertices.iter().map(|&vertex| block[vertex]))
        .collect();
    block.copy_from_slice(&ordered);
    let mut next = 0;
    for node in &found {
        let range = next..next + node.vertices.len();
        next = range.end;
        if node.children.is_empty() {
            reduce_fill(&mut block[range], local, (starts, rows), column_of_row)?;
        }
    }
    Ok(Some(found))
}

/// Orders one diagonal `block` according to `dissecting` when provided, or by minimum degree
/// otherwise, returning the dissection tree nodes if dissected.
///
/// Uses `local` as scratch space following [`reduce_fill`].
fn order_block(
    block: &mut [usize],
    local: &mut [usize],
    (pattern, column_of_row): ((&[usize], &[usize]), &[usize]),
    dissecting: Option<Dissecting>,
) -> Result<Option<Vec<dissection::Found>>, LinearError> {
    let found = match dissecting.filter(|each| block.len() > each.interiors()) {
        Some(each @ Dissecting::Every(_)) => {
            dissect(block, local, pattern, column_of_row, each.interiors())?
        }
        Some(each @ Dissecting::Cheaper(_)) => {
            return cheaper(block, local, pattern, column_of_row, each.interiors());
        }
        None => None,
    };
    if found.is_none() {
        reduce_fill(block, local, pattern, column_of_row)?;
    }
    Ok(found)
}

/// Orders one diagonal `block` by whichever of nested dissection (up to `most` columns per
/// interior via [`dissect`]) or minimum degree ([`reduce_fill`]) requires fewer elimination
/// multiply-adds according to [`multiply_adds`].
///
/// Returns the dissection tree nodes when nested dissection wins, or `None` when minimum
/// degree wins, ties, or the block does not split. Ties favor single-task minimum degree
/// factorization. Uses `local` as scratch space following [`reduce_fill`].
fn cheaper(
    block: &mut [usize],
    local: &mut [usize],
    pattern: (&[usize], &[usize]),
    column_of_row: &[usize],
    most: usize,
) -> Result<Option<Vec<dissection::Found>>, LinearError> {
    let mut fewest = block.to_vec();
    reduce_fill(&mut fewest, local, pattern, column_of_row)?;
    if let Some(found) = dissect(block, local, pattern, column_of_row, most)?
        && multiply_adds(block, local, pattern, column_of_row)
            < multiply_adds(&fewest, local, pattern, column_of_row)
    {
        return Ok(Some(found));
    }
    block.copy_from_slice(&fewest);
    Ok(None)
}

/// Counts the elimination multiply-adds for one diagonal `block` in its current column order,
/// pivoting each column on its matched row.
///
/// The count sums the squared number of subdiagonal entries in each column of the symbolic
/// factor of the symmetrized block pattern, since each elimination step multiplies every
/// subdiagonal `L` entry by every superdiagonal `U` entry. The symbolic factor is computed via
/// the elimination tree (Liu, "The role of elimination trees in sparse factorization", SIAM
/// Journal on Matrix Analysis and Applications 11(1), 1990) and row subtrees in time
/// proportional to the factor entries. Uses `local` as scratch space following
/// [`reduce_fill`].
fn multiply_adds(
    block: &[usize],
    local: &mut [usize],
    (starts, rows): (&[usize], &[usize]),
    column_of_row: &[usize],
) -> u128 {
    let size = block.len();
    index_block(block, local);
    // Each position's neighbors before it, in the pattern plus its transpose.
    let mut earlier: Vec<Vec<usize>> = vec![Vec::new(); size];
    for (position, &column) in block.iter().enumerate() {
        for other in block_neighbors(block, local, (starts, rows), column_of_row, column) {
            match other.cmp(&position) {
                std::cmp::Ordering::Less => earlier[position].push(other),
                std::cmp::Ordering::Greater => earlier[other].push(position),
                std::cmp::Ordering::Equal => {}
            }
        }
    }
    // The elimination tree, by Liu's algorithm with path compression.
    let (mut parent, mut ancestor) = (vec![NONE; size], vec![NONE; size]);
    for (position, before) in earlier.iter().enumerate() {
        for &first in before {
            let mut at = first;
            while at != NONE && at != position {
                let next = ancestor[at];
                ancestor[at] = position;
                if next == NONE {
                    parent[at] = position;
                }
                at = next;
            }
        }
    }
    // Each row's entries of `L` are the nodes of its subtree: the paths up the tree from
    // its neighbors before it.
    let (mut below, mut mark) = (vec![0_u128; size], vec![NONE; size]);
    for (position, before) in earlier.iter().enumerate() {
        mark[position] = position;
        for &first in before {
            let mut at = first;
            while at != NONE && mark[at] != position {
                below[at] += 1;
                mark[at] = position;
                at = parent[at];
            }
        }
    }
    below.iter().map(|entries| entries * entries).sum()
}

/// Reorders one diagonal `block` by approximate minimum degree on the symmetrized block
/// pattern, mapping each row to its matched column.
///
/// Uses `local` (one index per pattern column) to store each column's local index within
/// `block` while processing `(starts, rows)`.
fn reduce_fill(
    block: &mut [usize],
    local: &mut [usize],
    (starts, rows): (&[usize], &[usize]),
    column_of_row: &[usize],
) -> Result<(), LinearError> {
    // Every ordering of two columns or fewer produces identical fill.
    let size = block.len();
    if size <= 2 {
        return Ok(());
    }
    index_block(block, local);
    let mut block_starts = Vec::with_capacity(size + 1);
    let mut block_rows = Vec::new();
    block_starts.push(0);
    for &column in block.iter() {
        let first = block_rows.len();
        let neighbors = block_neighbors(block, local, (starts, rows), column_of_row, column);
        block_rows.extend(neighbors);
        block_rows[first..].sort_unstable();
        block_starts.push(block_rows.len());
    }
    let structure =
        SymbolicSparseColMatRef::new_checked(size, size, &block_starts, None, &block_rows);
    let (mut permutation, mut inverse) = (vec![0; size], vec![0; size]);
    // Allocate AMD workspace for off-diagonal symmetrized entries, 20% headroom, and one
    // index per column. Adding `size` to the entry count covers the per-column slots under
    // faer's 2.4-index-per-entry scratch formula.
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

    /// Builds the symmetric adjacency pattern of a `side` × `side` resistor grid.
    fn grid(side: usize) -> Arc<Pattern> {
        let at = |row: usize, column: usize| row * side + column;
        let mut positions = Vec::new();
        for row in 0..side {
            for column in 0..side {
                positions.push((at(row, column), at(row, column)));
                for (other, inside) in [
                    (at(row, column) + side, row + 1 < side),
                    (at(row, column) + 1, column + 1 < side),
                ] {
                    if inside {
                        positions.extend([(at(row, column), other), (other, at(row, column))]);
                    }
                }
            }
        }
        pattern(side * side, &positions)
    }

    /// Builds the symmetric adjacency pattern of a `length`-node path graph.
    fn path(length: usize) -> Arc<Pattern> {
        let mut positions: Vec<(usize, usize)> = (0..length).map(|node| (node, node)).collect();
        for node in 1..length {
            positions.extend([(node - 1, node), (node, node - 1)]);
        }
        pattern(length, &positions)
    }

    #[test]
    fn dissects_a_block_only_where_it_costs_fewer_multiply_adds() {
        // Verify the exact multiply-add thresholds documented on `Dissecting::Cheaper`.
        let dissections = |pattern| {
            Ordering::dissected(pattern, Dissecting::Cheaper(157))
                .expect("an ordering")
                .dissections()
                .len()
        };
        assert_eq!(dissections(grid(60)), 0);
        assert_eq!(dissections(grid(70)), 1);
        assert_eq!(dissections(path(5_000)), 0);
    }

    #[test]
    fn counts_the_multiply_adds_of_an_elimination() {
        // A path in natural order incurs zero fill: each column except the last holds one
        // subdiagonal entry and contributes one multiply-add. Eliminating the middle node
        // first touches both neighbors (four multiply-adds) and connects them, leaving one
        // subdiagonal entry for each remaining column except the last.
        let pattern = path(5);
        let structure = pattern.structure();
        let identity: Vec<usize> = (0..5).collect();
        let mut local = vec![NONE; 5];
        let count = |order: &[usize], local: &mut [usize]| {
            multiply_adds(
                order,
                local,
                (structure.col_ptr(), structure.row_idx()),
                &identity,
            )
        };
        assert_eq!(count(&identity, &mut local), 4);
        assert_eq!(count(&[2, 0, 1, 3, 4], &mut local), 4 + 1 + 1 + 1);
    }

    /// Builds a `dimension` × `dimension` [`Pattern`] containing every entry in `positions`.
    fn pattern(dimension: usize, positions: &[(usize, usize)]) -> Arc<Pattern> {
        let mut builder = PatternBuilder::new();
        for &(row, column) in positions {
            builder.entry(row, column);
        }
        Arc::new(builder.finish(dimension))
    }

    /// Computes the maximum transversal row matching for `pattern`, or returns undetermined
    /// columns.
    fn matching(pattern: &Pattern) -> Result<Vec<usize>, Vec<usize>> {
        let structure = pattern.structure();
        maximum_transversal(
            pattern.dimension(),
            structure.col_ptr(),
            structure.row_idx(),
        )
    }

    /// Returns `true` when a row permutation places an entry of `positions` on every diagonal
    /// position, checked by backtracking.
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
        // A node with a conductance to the reference and a voltage source: the source current
        // (column 1) appears only in the node equation, so the node voltage (column 0) takes
        // the source equation (row 1).
        let source = pattern(2, &[(0, 0), (0, 1), (1, 0)]);
        assert_eq!(matching(&source), Ok(vec![1, 0]));
        // A full diagonal is preserved regardless of off-diagonal entries.
        let full = pattern(3, &[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (1, 2), (2, 2)]);
        assert_eq!(matching(&full), Ok(vec![0, 1, 2]));
        // Two parallel voltage sources constrain the same node: their branch currents (columns
        // 1 and 2) appear only in that node equation.
        let parallel = pattern(3, &[(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)]);
        assert_eq!(matching(&parallel), Err(vec![1, 2]));
        let ordering = Ordering::new(parallel).expect("fits");
        assert!(ordering.is_singular());
        assert_eq!(ordering.undetermined(), [1, 2]);
    }

    #[test]
    fn splits_a_grounded_source_into_blocks_of_its_own() {
        // Nodes 0 and 1 share a conductance, and a source drives node 1 with current unknown 2.
        // Node 1's voltage solves first from the source equation, and the source current solves
        // last from node 1's equation.
        let held = pattern(3, &[(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]);
        let ordering = Ordering::new(held).expect("fits");
        assert_eq!(ordering.blocks(), 3);
        assert_eq!(ordering.columns(), [2, 0, 1]);
        assert_eq!(ordering.rows(), [1, 0, 2]);
    }

    proptest! {
        /// Verifies that the matching covers every column whenever a perfect matching exists
        /// and otherwise returns a Hall witness spanning fewer rows than columns.
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
            prop_assert_eq!(found.is_ok(), perfect_matching_exists(dimension, &positions));
            match found {
                Ok(rows) => {
                    let mut taken = vec![false; dimension];
                    for (column, &row) in rows.iter().enumerate() {
                        prop_assert!(positions.contains(&(row, column)));
                        prop_assert!(!taken[row]);
                        taken[row] = true;
                    }
                }
                Err(columns) => {
                    prop_assert!(columns.is_sorted() && !columns.is_empty());
                    let rows: std::collections::BTreeSet<usize> = positions
                        .iter()
                        .filter(|(_, column)| columns.binary_search(column).is_ok())
                        .map(|&(row, _)| row)
                        .collect();
                    prop_assert!(rows.len() < columns.len());
                }
            }
        }

        /// Verifies that the ordered matrix is block upper triangular with a stored diagonal
        /// and contains one diagonal block per strongly connected component.
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
