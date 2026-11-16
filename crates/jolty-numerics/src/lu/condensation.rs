//! The static condensation of a pattern's matrices: some of their unknowns eliminated, each
//! by its own pivot, before the sparse factorization of the rest, and recovered after each
//! solve.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use super::{LinearError, Scalar};
use crate::{Entry, Pattern, PatternBuilder};

/// No index: a row or a column eliminated, or one kept.
const NONE: usize = usize::MAX;

/// Which unknowns of a pattern's matrices are eliminated before the sparse factorization,
/// each by its pivot, and the pattern of the unknowns kept.
///
/// Eliminating the unknown `e` by the pivot `p = A_re` of the row `r`, its own or another's,
/// subtracts `A_ke/p` times the row `r` from every other row `k` with an entry in the column
/// `e`, and leaves out the row `r` and the column `e`: one step of Gaussian elimination
/// with a pivot fixed in advance. The eliminations, in turn, leave the kept unknowns a Schur
/// complement, which the sparse factorization factors; a system `A x = b` is then solved as
/// the kept unknowns' system with the right-hand side the same steps make of `b`, and each
/// unknown eliminated is recovered from its pivot's row, the last first: the static
/// condensation of finite elements (Wilson, "The static condensation algorithm",
/// International Journal for Numerical Methods in Engineering 8(1), 1974). A pivot off the
/// diagonal leaves its row `e` among the kept equations in the place of the unknown `r`.
///
/// The pivots given are those their matrices' makers prove to be at least every other entry
/// of their columns, so that each passes the factorization's threshold without a search,
/// and each multiplier `A_ke/p` is at most 1. Of them, in increasing order of their columns,
/// one is eliminated only if its pivot is stored, its row and its column are no other's
/// eliminated before it, and its column is still as proven: every elimination before it
/// either left the column alone or moved the column's entry in the row it eliminated by,
/// times a multiplier of at most 1, into rows where the column held nothing, so that no
/// entry grew past the pivot. A pivot off the diagonal is taken only if its row is no other
/// pivot's column and its column no other pivot's row. The others stay among the unknowns
/// kept.
///
/// Every sum runs in one order: the unknowns eliminated in increasing order, each one's
/// column in increasing row and its row in increasing column.
pub struct Condensation {
    /// The pattern of the matrices condensed.
    full: Arc<Pattern>,
    /// The pattern of the kept unknowns' matrix: the full pattern's entries among them, and
    /// those the eliminations fill.
    kept: Arc<Pattern>,
    /// The full pattern's unknown of each kept unknown, in increasing order.
    unknowns: Vec<usize>,
    /// The full pattern's row of each kept unknown's equation: its own, or the row of an
    /// unknown eliminated by another's row.
    rows: Vec<usize>,
    /// The full matrix's slot of each value the eliminations write, or [`NONE`] for one
    /// only they fill: the kept unknowns' matrix, then each entry of a row or a column
    /// eliminated later that an elimination writes to first.
    sources: Vec<usize>,
    /// Each unknown eliminated, in increasing order.
    eliminated: Vec<Eliminated>,
    /// The entries of each eliminated unknown's column but its pivot, unknown after
    /// unknown, each one's in increasing row: their rows and their places among the values
    /// worked on.
    column: Vec<Reference>,
    /// The entries of each eliminated unknown's pivot row but its pivot, unknown after
    /// unknown, each one's in increasing column: their columns and their places among the
    /// values worked on.
    row: Vec<Reference>,
    /// Where among the values the eliminations write each product of an elimination goes:
    /// unknown after unknown, for each entry of its column, for each entry of its row.
    updates: Vec<usize>,
    /// The eliminations whose pivots' rows, and those whose unknowns, an elimination before
    /// reads, by their indices, in increasing order.
    shared: (Vec<usize>, Vec<usize>),
}

/// An entry of an eliminated unknown's column or row, as an elimination reads it: where a
/// solve keeps the value of its row, or of its column, and where its own value lies.
#[derive(Debug, Clone, Copy)]
struct Reference {
    /// Where a solve keeps the value of the entry's row, for an entry of a column, or of
    /// its column, for an entry of a row: a kept unknown's index, or the number of kept
    /// unknowns and the elimination's index among the eliminated, for a pivot's row or an
    /// eliminated unknown.
    solved: usize,
    /// Where its value lies.
    at: At,
}

/// Where an elimination reads a value: at its slot in the full matrix, where no elimination
/// before writes it, as most are, or, with [`WRITTEN`] set, among the values the
/// eliminations write.
#[derive(Debug, Clone, Copy)]
struct At(usize);

/// The bit of an [`At`] that says it lies among the values the eliminations write: no slot
/// reaches it, as no vector holds that many values.
const WRITTEN: usize = 1 << (usize::BITS - 1);

/// An elimination as it is laid out: its unknown, its pivot's row and where its pivot
/// lies, where its column's and its row's entries lie, and where its updates begin.
type Laid = ((usize, usize, Place), (usize, usize), (usize, usize), usize);

/// The eliminations `laid` out, each pivot where `resolve` puts it, each flagged by the
/// entries of `column` and `row`, every elimination's, of `count` kept unknowns: whether one
/// before it reads its pivot's row or its unknown, and whether it reads the full matrix
/// alone.
fn eliminations(
    laid: Vec<Laid>,
    (column, row): (&[Reference], &[Reference]),
    count: usize,
    resolve: impl Fn(Place) -> At,
) -> Vec<Eliminated> {
    let mut shared = vec![(false, false); laid.len()];
    for entry in column {
        if let Some(later) = entry.solved.checked_sub(count) {
            shared[later].0 = true;
        }
    }
    for entry in row {
        if let Some(later) = entry.solved.checked_sub(count) {
            shared[later].1 = true;
        }
    }
    laid.into_iter()
        .zip(shared)
        .map(
            |(((unknown, row_of, pivot), columns, entries, updates), shared)| {
                let pivot = resolve(pivot);
                let simple = [pivot]
                    .into_iter()
                    .chain(column[columns.0..columns.1].iter().map(|entry| entry.at))
                    .chain(row[entries.0..entries.1].iter().map(|entry| entry.at))
                    .all(|at| at.0 & WRITTEN == 0);
                Eliminated {
                    unknown,
                    row: row_of,
                    pivot,
                    column: columns,
                    entries,
                    updates,
                    shared,
                    simple,
                }
            },
        )
        .collect()
}

/// Where an entry lies while the eliminations are laid out: in the full matrix at its
/// slot, at a claim among the kept entries, or after them, among those of rows and columns
/// eliminated later.
#[derive(Debug, Clone, Copy)]
enum Place {
    Full(usize),
    Kept(usize),
    Later(usize),
}

impl At {
    /// The value at this place, of the full matrix's `values` and the `written` ones.
    #[inline]
    fn of<T: Copy>(self, values: &[T], written: &[T]) -> T {
        if self.0 & WRITTEN == 0 {
            values[self.0]
        } else {
            written[self.0 & !WRITTEN]
        }
    }
}

/// How an elimination reads its values: [`Full`], from the full matrix alone, for one that
/// reads nothing an elimination before it wrote, or [`Either`], from wherever each lies.
trait Read {
    /// The value at `at`, of the full matrix's `values` and the `written` ones.
    fn read<T: Copy>(at: At, values: &[T], written: &[T]) -> T;
}

/// Reads the full matrix alone.
struct Full;

impl Read for Full {
    #[inline]
    fn read<T: Copy>(at: At, values: &[T], _: &[T]) -> T {
        values[at.0]
    }
}

/// Reads the full matrix or the values written, as each [`At`] says.
struct Either;

impl Read for Either {
    #[inline]
    fn read<T: Copy>(at: At, values: &[T], written: &[T]) -> T {
        at.of(values, written)
    }
}

/// One unknown eliminated: where its pivot, its column, its row and its updates lie.
#[derive(Debug, Clone, Copy)]
struct Eliminated {
    /// The unknown, in the full pattern: its column.
    unknown: usize,
    /// The row of its pivot.
    row: usize,
    /// Where its pivot lies.
    pivot: At,
    /// Where its column's entries lie among [`Condensation::column`].
    column: (usize, usize),
    /// Where its row's entries lie among [`Condensation::row`].
    entries: (usize, usize),
    /// Where its updates begin among [`Condensation::updates`].
    updates: usize,
    /// Whether an elimination before it reads the value of its pivot's row, as a column's
    /// entry there, and of its unknown, as a row's entry there: a solve keeps what it makes
    /// of those after the kept unknowns' values, and else reads and writes the right-hand
    /// side alone, as an unknown that shares nothing with another needs.
    shared: (bool, bool),
    /// Whether every value it reads, its pivot and the other entries of its column and its
    /// row, is the full matrix's, which no elimination before it wrote, as for an unknown
    /// that shares nothing with another eliminated.
    simple: bool,
}

impl Eliminated {
    /// The range of its column's entries.
    const fn column(&self) -> std::ops::Range<usize> {
        self.column.0..self.column.1
    }

    /// The range of its row's entries.
    const fn entries(&self) -> std::ops::Range<usize> {
        self.entries.0..self.entries.1
    }
}

/// What a factorization of a condensed matrix keeps for its solves: the values the
/// eliminations wrote, the kept unknowns' matrix first, each eliminated unknown's pivot,
/// the multipliers of its column, each entry over the pivot, and the entries of its row,
/// in the order [`Condensation`] lists them.
#[derive(Debug, Clone)]
pub(super) struct Saved<T> {
    written: Vec<T>,
    pivots: Vec<T>,
    multipliers: Vec<T>,
    row: Vec<T>,
}

impl<T> Saved<T> {
    /// Nothing saved yet.
    pub(super) const fn new() -> Self {
        Self {
            written: Vec::new(),
            pivots: Vec::new(),
            multipliers: Vec::new(),
            row: Vec::new(),
        }
    }
}

/// Where each unknown and each equation of a full pattern lies once some of its unknowns
/// are eliminated, each by its pivot's row, in turn.
struct Numbering {
    /// The kept index of each unknown, or [`NONE`] for one eliminated.
    index: Vec<usize>,
    /// The full pattern's unknown of each kept one, in increasing order.
    unknowns: Vec<usize>,
    /// The kept index of each row: its own unknown's, or, for the row of an unknown
    /// eliminated by another's row, that row's unknown's; [`NONE`] for a pivot's row.
    row_index: Vec<usize>,
    /// The full pattern's row of each kept index.
    rows: Vec<usize>,
    /// The elimination each row is the pivot's row of, and each column the unknown of, or
    /// [`NONE`].
    eliminated: (Vec<usize>, Vec<usize>),
}

impl Numbering {
    /// The numbering of a full pattern of `dimension` unknowns once each of `taken`, a
    /// pivot's row and column, is eliminated.
    fn new(dimension: usize, taken: &[(usize, usize)]) -> Self {
        let (mut eliminated_row, mut eliminated_column) =
            (vec![NONE; dimension], vec![NONE; dimension]);
        for (index, &(row, column)) in taken.iter().enumerate() {
            eliminated_row[row] = index;
            eliminated_column[column] = index;
        }
        let mut index = vec![NONE; dimension];
        let mut unknowns = Vec::new();
        for unknown in (0..dimension).filter(|&unknown| eliminated_column[unknown] == NONE) {
            index[unknown] = unknowns.len();
            unknowns.push(unknown);
        }
        let mut row_index: Vec<usize> = (0..dimension)
            .map(|row| match eliminated_row[row] {
                NONE => index[row],
                _ => NONE,
            })
            .collect();
        for &(row, column) in taken.iter().filter(|&&(row, column)| row != column) {
            row_index[column] = index[row];
        }
        let mut rows = vec![NONE; unknowns.len()];
        for (row, &kept) in row_index.iter().enumerate() {
            if kept != NONE {
                rows[kept] = row;
            }
        }
        Self {
            index,
            unknowns,
            row_index,
            rows,
            eliminated: (eliminated_row, eliminated_column),
        }
    }

    /// Claims in `builder` the kept unknowns' entry at the full pattern's `row` and
    /// `column`, if both are kept.
    fn claim(&self, builder: &mut PatternBuilder, (row, column): (usize, usize)) -> Option<Entry> {
        let (row, column) = (self.row_index[row], self.index[column]);
        (row != NONE && column != NONE).then(|| builder.entry(row, column))
    }

    /// Where a solve keeps the value of `row`: its kept index, or after the kept unknowns,
    /// that of its elimination.
    fn solved_row(&self, row: usize) -> usize {
        match self.eliminated.0[row] {
            NONE => self.row_index[row],
            elimination => self.unknowns.len() + elimination,
        }
    }

    /// Where a solve keeps the value of `column`'s unknown, as [`Numbering::solved_row`].
    fn solved_column(&self, column: usize) -> usize {
        match self.eliminated.1[column] {
            NONE => self.index[column],
            elimination => self.unknowns.len() + elimination,
        }
    }
}

/// The entries of a pattern as eliminations change it: those of the full pattern, which
/// it searches by column, and those the eliminations fill, by row and by column.
struct Evolving<'a> {
    starts: &'a [usize],
    rows: &'a [usize],
    full: &'a Pattern,
    filled: HashSet<(usize, usize)>,
    filled_rows: HashMap<usize, Vec<usize>>,
    filled_columns: HashMap<usize, Vec<usize>>,
    /// Whether each row, and each column, has been eliminated.
    gone: (Vec<bool>, Vec<bool>),
}

impl<'a> Evolving<'a> {
    fn new(full: &'a Pattern) -> Self {
        let structure = full.structure();
        let dimension = full.dimension();
        Self {
            starts: structure.col_ptr(),
            rows: structure.row_idx(),
            full,
            filled: HashSet::new(),
            filled_rows: HashMap::new(),
            filled_columns: HashMap::new(),
            gone: (vec![false; dimension], vec![false; dimension]),
        }
    }

    /// The full pattern's slot at `row` and `column`, if it stores one.
    fn slot(&self, row: usize, column: usize) -> Option<usize> {
        let start = self.starts[column];
        self.rows[start..self.starts[column + 1]]
            .binary_search(&row)
            .ok()
            .map(|offset| start + offset)
    }

    /// Whether the pattern holds an entry at `row` and `column`, of its own or filled.
    fn holds(&self, row: usize, column: usize) -> bool {
        self.slot(row, column).is_some() || self.filled.contains(&(row, column))
    }

    /// The rows of `column`'s entries, but `pivot`'s and those eliminated, in increasing
    /// order.
    fn column(&self, column: usize, pivot: usize) -> Vec<usize> {
        let own = &self.rows[self.starts[column]..self.starts[column + 1]];
        let filled = self.filled_rows.get(&column).map_or(&[][..], Vec::as_slice);
        let mut rows: Vec<usize> = own
            .iter()
            .chain(filled)
            .copied()
            .filter(|&row| row != pivot && !self.gone.0[row])
            .collect();
        rows.sort_unstable();
        rows
    }

    /// The columns of `row`'s entries, but `pivot`'s and those eliminated, in increasing
    /// order.
    fn row(&self, row: usize, pivot: usize) -> Vec<usize> {
        let own = self.full.row(row).iter().map(|&(_, column)| column);
        let filled = self.filled_columns.get(&row).map_or(&[][..], Vec::as_slice);
        let mut columns: Vec<usize> = own
            .chain(filled.iter().copied())
            .filter(|&column| column != pivot && !self.gone.1[column])
            .collect();
        columns.sort_unstable();
        columns
    }

    /// Fills the entry at `row` and `column`.
    fn fill(&mut self, row: usize, column: usize) {
        self.filled.insert((row, column));
        self.filled_rows.entry(column).or_default().push(row);
        self.filled_columns.entry(row).or_default().push(column);
    }

    /// Eliminates the column `unknown` by the row `row`: fills each product of the other
    /// entries of its column and its row where the pattern held nothing, and returns the
    /// rows and the columns of those entries, and the places the products were added to an
    /// entry held before, in the order of the updates.
    fn eliminate(
        &mut self,
        (row, unknown): (usize, usize),
    ) -> (Vec<usize>, Vec<usize>, Vec<(usize, usize)>) {
        let rows = self.column(unknown, row);
        let columns = self.row(row, unknown);
        let mut added = Vec::new();
        for &target in &rows {
            for &along in &columns {
                if self.holds(target, along) {
                    added.push((target, along));
                } else {
                    self.fill(target, along);
                }
            }
        }
        self.gone.0[row] = true;
        self.gone.1[unknown] = true;
        (rows, columns, added)
    }
}

impl Condensation {
    /// The condensation of `full`'s matrices that eliminates those of the `pivots`, each a
    /// row and the column of an unknown, that it can, as [`Condensation`] says. The pivot of
    /// each is to be at least every other entry of its column in every matrix condensed.
    #[must_use]
    pub fn new(full: Arc<Pattern>, pivots: &[(usize, usize)]) -> Self {
        let dimension = full.dimension();
        let mut candidates: Vec<(usize, usize)> = pivots
            .iter()
            .copied()
            .filter(|&(row, column)| row < dimension && column < dimension)
            .collect();
        candidates.sort_unstable_by_key(|&(row, column)| (column, row));
        candidates.dedup();
        let (mut pivot_row, mut pivot_column) = (vec![false; dimension], vec![false; dimension]);
        for &(row, column) in &candidates {
            pivot_row[row] = true;
            pivot_column[column] = true;
        }
        // Each candidate in turn, eliminated when its column is still as proven.
        let mut evolving = Evolving::new(&full);
        let mut proven = pivot_column.clone();
        let mut taken = Vec::new();
        for &(row, column) in &candidates {
            let (rows_gone, columns_gone) = &evolving.gone;
            let paired = row == column || !(pivot_column[row] || pivot_row[column]);
            if rows_gone[row]
                || columns_gone[column]
                || !paired
                || !proven[column]
                || evolving.slot(row, column).is_none()
            {
                continue;
            }
            let (_, _, added) = evolving.eliminate((row, column));
            for (_, along) in added {
                proven[along] = false;
            }
            taken.push((row, column));
        }
        Self::of(full, &taken)
    }

    /// The condensation of `full`'s matrices that eliminates each of `taken`, a pivot's row
    /// and column, in their order.
    fn of(full: Arc<Pattern>, taken: &[(usize, usize)]) -> Self {
        let numbering = Numbering::new(full.dimension(), taken);
        let mut builder = PatternBuilder::new();
        // The full pattern's entries among the kept unknowns, in its order of slots, and
        // each entry the eliminations fill among them, with the full pattern's slot of each,
        // or `NONE` for one filled.
        let mut kept_entries: Vec<(Entry, usize)> = full
            .positions()
            .enumerate()
            .filter_map(|(slot, position)| Some((numbering.claim(&mut builder, position)?, slot)))
            .collect();
        // Where each entry an elimination writes lies among the values written: a kept
        // one's claim, or one of a row or a column eliminated later, in the order first
        // written, with the full pattern's slot of each, or `NONE`.
        let mut writes: HashMap<(usize, usize), Place> = HashMap::new();
        let mut later: Vec<usize> = Vec::new();
        let mut evolving = Evolving::new(&full);
        let (mut list, mut column_entries, mut row_entries, mut updates) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &(row, unknown) in taken {
            let rows = evolving.column(unknown, row);
            let columns = evolving.row(row, unknown);
            let first = (column_entries.len(), row_entries.len(), updates.len());
            // An entry no elimination before wrote is read from the full matrix.
            let read = |position: (usize, usize)| {
                writes.get(&position).copied().unwrap_or_else(|| {
                    Place::Full(evolving.slot(position.0, position.1).unwrap_or(NONE))
                })
            };
            let pivot = read((row, unknown));
            for &each in &rows {
                column_entries.push((numbering.solved_row(each), read((each, unknown))));
            }
            for &each in &columns {
                row_entries.push((numbering.solved_column(each), read((row, each))));
            }
            for &target in &rows {
                for &along in &columns {
                    let position = (target, along);
                    let place = *writes.entry(position).or_insert_with(|| {
                        let slot = evolving.slot(target, along).unwrap_or(NONE);
                        if let Some(entry) = numbering.claim(&mut builder, position) {
                            kept_entries.push((entry, slot));
                            Place::Kept(kept_entries.len() - 1)
                        } else {
                            later.push(slot);
                            Place::Later(later.len() - 1)
                        }
                    });
                    updates.push(place);
                }
            }
            evolving.eliminate((row, unknown));
            list.push((
                (unknown, row, pivot),
                (first.0, column_entries.len()),
                (first.1, row_entries.len()),
                first.2,
            ));
        }
        let kept = builder.finish(numbering.unknowns.len());
        let mut sources = vec![NONE; kept.len()];
        let claims: Vec<usize> = kept_entries
            .iter()
            .map(|&(entry, slot)| {
                let at = kept.slot(entry);
                sources[at] = slot;
                at
            })
            .collect();
        sources.extend(&later);
        let written = |place: Place| match place {
            Place::Full(slot) => slot,
            Place::Kept(claim) => claims[claim],
            Place::Later(offset) => kept.len() + offset,
        };
        let resolve = |place: Place| match place {
            Place::Full(slot) => At(slot),
            _ => At(written(place) | WRITTEN),
        };
        let reference = |(solved, place): (usize, Place)| Reference {
            solved,
            at: resolve(place),
        };
        let column: Vec<Reference> = column_entries.into_iter().map(reference).collect();
        let row: Vec<Reference> = row_entries.into_iter().map(reference).collect();
        let count = numbering.unknowns.len();
        let eliminated = eliminations(list, (&column, &row), count, resolve);
        let indices = |pick: fn(&Eliminated) -> bool| {
            (0..eliminated.len())
                .filter(|&index| pick(&eliminated[index]))
                .collect()
        };
        let shared = (indices(|each| each.shared.0), indices(|each| each.shared.1));
        Self {
            eliminated,
            shared,
            column,
            row,
            updates: updates.into_iter().map(written).collect(),
            full,
            kept: Arc::new(kept),
            unknowns: numbering.unknowns,
            rows: numbering.rows,
            sources,
        }
    }

    /// The pattern of the matrices condensed.
    #[must_use]
    pub fn full(&self) -> &Arc<Pattern> {
        &self.full
    }

    /// The pattern of the kept unknowns' matrix, which the sparse factorization factors.
    #[must_use]
    pub fn kept(&self) -> &Arc<Pattern> {
        &self.kept
    }

    /// The number of unknowns eliminated.
    #[must_use]
    pub fn eliminated(&self) -> usize {
        self.eliminated.len()
    }

    /// The full pattern's unknown of the kept unknown at `index`.
    pub(super) fn unknown(&self, index: usize) -> usize {
        self.unknowns[index]
    }

    /// Writes into `saved` what the eliminations make of the full matrix whose stored
    /// values are `values`, and what its solves need, and returns the kept unknowns'
    /// matrix.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] naming an eliminated unknown whose pivot is zero
    /// or not finite.
    pub(super) fn condense<'a, T: Scalar>(
        &self,
        values: &[T],
        saved: &'a mut Saved<T>,
    ) -> Result<&'a [T], LinearError> {
        let Saved {
            written,
            pivots,
            multipliers,
            row,
        } = saved;
        written.clear();
        written.extend(
            self.sources
                .iter()
                .map(|&slot| if slot == NONE { T::ZERO } else { values[slot] }),
        );
        pivots.clear();
        multipliers.clear();
        row.resize(self.row.len(), T::ZERO);
        let mut taken = (pivots, multipliers, row);
        for eliminated in &self.eliminated {
            if eliminated.simple {
                self.eliminate::<Full, T>(eliminated, (values, written), &mut taken)?;
            } else {
                self.eliminate::<Either, T>(eliminated, (values, written), &mut taken)?;
            }
        }
        Ok(&written[..self.kept.len()])
    }

    /// Eliminates `eliminated` from the full matrix's `values` and the `written` ones, as
    /// [`Condensation::condense`] does, each value read as `R` reads it, keeping its pivot,
    /// its column's multipliers and its row's entries in `taken`.
    #[inline]
    fn eliminate<R: Read, T: Scalar>(
        &self,
        eliminated: &Eliminated,
        (values, written): (&[T], &mut [T]),
        (pivots, multipliers, row): &mut (&mut Vec<T>, &mut Vec<T>, &mut Vec<T>),
    ) -> Result<(), LinearError> {
        let pivot = R::read(eliminated.pivot, values, written);
        if pivot.magnitude() == 0.0 || !pivot.is_finite() {
            return Err(LinearError::Singular {
                columns: vec![eliminated.unknown],
            });
        }
        pivots.push(pivot);
        let range = eliminated.entries();
        let along = &mut row[range.clone()];
        for (value, entry) in along.iter_mut().zip(&self.row[range]) {
            *value = R::read(entry.at, values, written);
        }
        let mut updates = self.updates[eliminated.updates..].iter();
        for entry in &self.column[eliminated.column()] {
            let multiplier = R::read(entry.at, values, written) / pivot;
            multipliers.push(multiplier);
            for (&along, &slot) in along.iter().zip(&mut updates) {
                written[slot] -= multiplier * along;
            }
        }
        Ok(())
    }

    /// The number of values a solve keeps, as [`Condensation::reduce`] and
    /// [`Condensation::expand`] read and write them: one per kept unknown, then one per
    /// unknown eliminated.
    pub(super) fn solved(&self) -> usize {
        self.unknowns.len() + self.eliminated.len()
    }

    /// Writes into `into`, of [`Condensation::solved`] values, the right-hand side of the
    /// kept unknowns' system of the full system whose right-hand side is `rhs`, one value per
    /// kept unknown, as the eliminations make it of `b`, and after them what each
    /// elimination left of its pivot's row's value; or, when `transposed`, the right-hand
    /// side of the transposed kept system, each pivot's row `r` taking each other column
    /// `c` of the row's `A_rc/p` times its unknown's value, and after them what each left of
    /// its unknown's.
    pub(super) fn reduce<T: Scalar>(
        &self,
        saved: &Saved<T>,
        (rhs, transposed): (&[T], bool),
        into: &mut [T],
    ) {
        let count = self.unknowns.len();
        let kept_indices = if transposed {
            &self.unknowns
        } else {
            &self.rows
        };
        for (into, &index) in into.iter_mut().zip(kept_indices) {
            *into = rhs[index];
        }
        // The value each elimination takes, of its unknown or of its pivot's row: kept after
        // the kept unknowns' where an elimination before it changes it, and else the
        // right-hand side's.
        let taken = |each: &Eliminated| {
            if transposed {
                (each.unknown, each.shared.1)
            } else {
                (each.row, each.shared.0)
            }
        };
        let shared = if transposed {
            &self.shared.1
        } else {
            &self.shared.0
        };
        for &index in shared {
            into[count + index] = rhs[taken(&self.eliminated[index]).0];
        }
        let ranges = self.eliminated.iter().zip(&saved.pivots).enumerate();
        for (index, (eliminated, &pivot)) in ranges {
            let value = match taken(eliminated) {
                (_, true) => into[count + index],
                (row, false) => rhs[row],
            };
            if transposed {
                let range = eliminated.entries();
                for (entry, &along) in self.row[range.clone()].iter().zip(&saved.row[range]) {
                    into[entry.solved] -= along / pivot * value;
                }
            } else {
                let range = eliminated.column();
                let (entries, multipliers) =
                    (&self.column[range.clone()], &saved.multipliers[range]);
                for (entry, &multiplier) in entries.iter().zip(multipliers) {
                    into[entry.solved] -= multiplier * value;
                }
            }
        }
    }

    /// Writes into `rhs` the solution of the full system from `solved`, the kept unknowns'
    /// solution and what `reduce` kept beyond it, each eliminated unknown recovered from its
    /// pivot's row, the last first: `(b_r − Σ A_rc x_c)/p`, or, when `transposed`,
    /// `b_e/p − Σ (A_ke/p) x_k` for the row `r` of the pivot, where `b` is still `rhs`'s
    /// value for a row or an unknown `reduce` kept nothing of.
    pub(super) fn expand<T: Scalar>(
        &self,
        saved: &Saved<T>,
        (solved, transposed): (&mut [T], bool),
        rhs: &mut [T],
    ) {
        let count = self.unknowns.len();
        let ranges = self.eliminated.iter().zip(&saved.pivots).enumerate().rev();
        for (index, (eliminated, &pivot)) in ranges {
            // What `reduce` took, and whether it kept it; then where the value recovered
            // goes, and whether an elimination before it reads it.
            let (row, unknown) = eliminated.shared;
            let ((taken, reduced), (recovered, read)) = if transposed {
                ((eliminated.unknown, unknown), (eliminated.row, row))
            } else {
                ((eliminated.row, row), (eliminated.unknown, unknown))
            };
            let given = if reduced {
                solved[count + index]
            } else {
                rhs[taken]
            };
            let value = if transposed {
                let range = eliminated.column();
                let (entries, multipliers) =
                    (&self.column[range.clone()], &saved.multipliers[range]);
                entries
                    .iter()
                    .zip(multipliers)
                    .fold(given / pivot, |value, (entry, &multiplier)| {
                        value - multiplier * solved[entry.solved]
                    })
            } else {
                let range = eliminated.entries();
                self.row[range.clone()]
                    .iter()
                    .zip(&saved.row[range])
                    .fold(given, |value, (entry, &along)| {
                        value - along * solved[entry.solved]
                    })
                    / pivot
            };
            if read {
                solved[count + index] = value;
            }
            rhs[recovered] = value;
        }
        let kept_indices = if transposed {
            &self.rows
        } else {
            &self.unknowns
        };
        for (&index, &value) in kept_indices.iter().zip(&solved[..count]) {
            rhs[index] = value;
        }
    }
}

impl fmt::Debug for Condensation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Condensation")
            .field("unknowns", &self.full.dimension())
            .field("eliminated", &self.eliminated.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lu::{Ordering, SparseLu};

    /// The pattern with an entry at each of `positions`, of `dimension` unknowns.
    fn pattern(dimension: usize, positions: &[(usize, usize)]) -> Arc<Pattern> {
        let mut builder = PatternBuilder::new();
        for &(row, column) in positions {
            builder.entry(row, column);
        }
        Arc::new(builder.finish(dimension))
    }

    #[test]
    fn eliminates_an_unknown_into_the_fill_of_its_neighbours() {
        // Unknown 2 joins 0 and 1, which share no entry: eliminating it fills (0, 1) and
        // (1, 0).
        let full = pattern(3, &[(0, 0), (1, 1), (2, 2), (0, 2), (2, 1)]);
        let condensation = Condensation::new(full, &[(2, 2)]);
        assert_eq!(condensation.eliminated(), 1);
        let kept: Vec<_> = condensation.kept().positions().collect();
        assert_eq!(kept, [(0, 0), (0, 1), (1, 1)]);
    }

    #[test]
    fn keeps_an_unknown_whose_column_an_elimination_added_to() {
        // Eliminating 1 adds its row's (1, 2) times its column's (0, 1) to (0, 2), an entry
        // of 2's column, which 2's proof bounds no longer: 2 stays. 3, alone in its row and
        // column but for 0, goes.
        let full = pattern(
            4,
            &[
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (0, 1),
                (1, 2),
                (0, 2),
                (2, 0),
                (0, 3),
                (3, 0),
            ],
        );
        let condensation = Condensation::new(full, &[(1, 1), (2, 2), (3, 3)]);
        assert_eq!(condensation.eliminated(), 2);
        assert_eq!(condensation.kept().dimension(), 2);
        // An unknown without a stored pivot stays too.
        let unpivoted = Condensation::new(pattern(2, &[(0, 0), (0, 1), (1, 0)]), &[(1, 1)]);
        assert_eq!(unpivoted.eliminated(), 0);
    }

    #[test]
    fn eliminates_a_series_current_by_its_inner_nodes_row() {
        // A terminal 0 of conductance 1, an inner node 1 of conductance 2, and the current
        // 2 through 0.5 ohm between them: the current goes by the inner node's row, whose
        // entry is −1, and the current's row takes the inner node's place.
        let positions = [
            ((0, 0), 1.0),
            ((0, 2), 1.0),
            ((1, 1), 2.0),
            ((1, 2), -1.0),
            ((2, 0), 1.0),
            ((2, 1), -1.0),
            ((2, 2), -0.5),
        ];
        let mut builder = PatternBuilder::new();
        let entries: Vec<_> = positions
            .iter()
            .map(|&((row, column), _)| builder.entry(row, column))
            .collect();
        let full = Arc::new(builder.finish(3));
        let mut values = vec![0.0; full.len()];
        for (&entry, &(_, value)) in entries.iter().zip(&positions) {
            values[full.slot(entry)] = value;
        }
        let condensation = Arc::new(Condensation::new(Arc::clone(&full), &[(1, 2)]));
        assert_eq!(condensation.eliminated(), 1);
        let ordering = Ordering::condensed(condensation, None).expect("fits");
        assert_eq!(ordering.factored().dimension(), 2);
        let mut lu = SparseLu::new(Arc::new(ordering));
        let mut factors = lu.factor(&values).expect("regular");
        // x = (2, 1, −1), and y = (1, 1, 2) of the transpose, every step exact.
        let mut x = [1.0, 3.0, 1.5];
        factors.solve(&mut x).expect("finite");
        assert_eq!(x, [2.0, 1.0, -1.0]);
        let mut y = [3.0, 0.0, -1.0];
        factors.solve_transpose(&mut y).expect("finite");
        assert_eq!(y, [1.0, 1.0, 2.0]);
    }

    #[test]
    fn names_an_eliminated_unknown_whose_pivot_is_zero() {
        let full = pattern(2, &[(0, 0), (1, 1), (0, 1), (1, 0)]);
        let condensation = Arc::new(Condensation::new(Arc::clone(&full), &[(1, 1)]));
        let ordering = Ordering::condensed(condensation, None).expect("fits");
        let mut lu = SparseLu::new(Arc::new(ordering));
        // The values by slot, column after column: (0, 0), (1, 0), (0, 1), (1, 1).
        let singular = lu.factor(&[2.0, 1.0, 1.0, 0.0]).map(|_| ());
        assert_eq!(singular, Err(LinearError::Singular { columns: vec![1] }));
        let mut x = [3.0, 1.0];
        lu.factor(&[2.0, 1.0, 1.0, 1.0])
            .and_then(|mut factors| factors.solve(&mut x))
            .expect("regular");
        // 2a + b = 3 and a + b = 1: a = 2, b = −1, exactly.
        assert_eq!(x, [2.0, -1.0]);
    }
}
