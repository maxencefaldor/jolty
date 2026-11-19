//! Static condensation of sparse circuit matrices, eliminating pre-selected unknowns by fixed
//! pivots before sparse factorization and recovering them after each solve.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use super::{LinearError, NONE, Scalar};
use crate::{Entry, Pattern, PatternBuilder};

/// Symbolic static-condensation plan identifying eliminated unknowns, their pivots, and the
/// reduced Schur-complement [`Pattern`] of retained unknowns.
///
/// Eliminating unknown `e` using pivot `p = A_re` subtracts `A_ke / p` times row `r` from
/// every other row `k` with an entry in column `e`, omitting row `r` and column `e`. Repeating
/// this step forms a Schur complement on the retained unknowns, matching finite-element static
/// condensation (Wilson, "The static condensation algorithm", International Journal for
/// Numerical Methods in Engineering 8(1), 1974). An off-diagonal pivot places row `e` in the
/// retained equation slot of unknown `r`.
///
/// Candidate pivots are assumed to dominate every other entry in their columns so multipliers
/// stay bounded by 1 without runtime pivot search. Candidates are accepted in ascending column
/// order only if the pivot is stored, its row and column remain uneliminated, and prior
/// eliminations did not touch existing entries in that column (leaving dominance intact).
/// Off-diagonal pivots are accepted only when their row and column do not appear as another
/// candidate's column or row; remaining unknowns stay in the kept system.
///
/// All accumulations execute in a fixed order: ascending eliminated column, ascending column
/// row index, and ascending pivot-row column index.
pub struct Condensation {
    /// Original uncondensed matrix pattern.
    full: Arc<Pattern>,
    /// Reduced Schur-complement pattern over retained unknowns (original plus fill entries).
    kept: Arc<Pattern>,
    /// Full-system unknown index for each retained unknown, in ascending order.
    unknowns: Vec<usize>,
    /// Full-system equation row index corresponding to each retained unknown.
    rows: Vec<usize>,
    /// Full-matrix slot for each written value (or [`NONE`] for pure fill), starting with the
    /// kept matrix followed by later-eliminated intermediates.
    sources: Vec<usize>,
    /// Accepted elimination steps in ascending eliminated-column order.
    eliminated: Vec<Eliminated>,
    /// Non-pivot entries of each eliminated column (ascending row), concatenated across steps.
    column: Vec<Reference>,
    /// Non-pivot entries of each eliminated pivot row (ascending column), concatenated across
    /// steps.
    row: Vec<Reference>,
    /// Target write-buffer indices for outer-product updates, concatenated across steps.
    updates: Vec<usize>,
    /// Elimination indices whose pivot rows and columns are read by earlier steps, sorted.
    shared: (Vec<usize>, Vec<usize>),
}

/// Entry reference within an eliminated column or row, pairing its solve-buffer index with its
/// matrix value location.
#[derive(Debug, Clone, Copy)]
struct Reference {
    /// Index in the solve buffer: a retained unknown index, or `kept_count + step_index` for a
    /// shared eliminated row or unknown.
    solved: usize,
    /// Storage address of the matrix entry value.
    at: At,
}

/// Matrix entry address pointing either to a full-matrix slot or (when [`WRITTEN`] is set) to
/// the elimination write buffer.
#[derive(Debug, Clone, Copy)]
struct At(usize);

/// High-bit tag on [`At`] marking an entry stored in the elimination write buffer rather than
/// the input matrix.
///
/// Derivation: high bit `1 << (usize::BITS - 1)` of `usize`, disjoint from valid Rust `Vec`
/// indices (bounded by `isize::MAX`).
const WRITTEN: usize = 1 << (usize::BITS - 1);

/// Intermediate symbolic layout for one elimination step before final slot resolution.
struct Laid {
    unknown: usize,
    row: usize,
    pivot: Place,
    column: (usize, usize),
    entries: (usize, usize),
    updates: usize,
}

/// Accepted elimination step recording its pivot `(row, unknown)` and sorted active column
/// rows and row columns at elimination time.
struct Step {
    row: usize,
    unknown: usize,
    rows: Vec<usize>,
    columns: Vec<usize>,
}

/// Returns the slot of `(row, column)` in the compressed sparse column pattern described by
/// `starts` and `rows`, if stored.
fn slot_in(starts: &[usize], rows: &[usize], row: usize, column: usize) -> Option<usize> {
    let start = starts[column];
    rows[start..starts[column + 1]]
        .binary_search(&row)
        .ok()
        .map(|offset| start + offset)
}

/// Finalizes laid-out eliminations by resolving pivot addresses and flagging shared
/// solve-buffer dependencies and input-only (`simple`) steps.
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
        .map(|(item, shared)| {
            let pivot = resolve(item.pivot);
            let simple = [pivot]
                .into_iter()
                .chain(
                    column[item.column.0..item.column.1]
                        .iter()
                        .map(|entry| entry.at),
                )
                .chain(
                    row[item.entries.0..item.entries.1]
                        .iter()
                        .map(|entry| entry.at),
                )
                .all(|at| at.0 & WRITTEN == 0);
            Eliminated {
                unknown: item.unknown,
                row: item.row,
                pivot,
                column: item.column,
                entries: item.entries,
                updates: item.updates,
                shared,
                simple,
            }
        })
        .collect()
}

/// Pre-resolution entry location: a full-matrix slot, a claimed kept-pattern entry, or an
/// intermediate entry eliminated by a later step.
#[derive(Debug, Clone, Copy)]
enum Place {
    Full(usize),
    Kept(usize),
    Later(usize),
}

impl At {
    /// Reads the scalar at this address from either `values` or `written`.
    #[inline]
    fn of<T: Copy>(self, values: &[T], written: &[T]) -> T {
        if self.0 & WRITTEN == 0 {
            values[self.0]
        } else {
            written[self.0 & !WRITTEN]
        }
    }
}

/// Strategy for reading matrix values during condensation: [`Full`] for input-only steps and
/// [`Either`] when prior steps may have written updated values.
trait Read {
    /// Reads the value addressed by `at` from `values` or `written`.
    fn read<T: Copy>(at: At, values: &[T], written: &[T]) -> T;
}

/// Value reader accessing the unmodified input matrix directly.
struct Full;

impl Read for Full {
    #[inline]
    fn read<T: Copy>(at: At, values: &[T], _: &[T]) -> T {
        values[at.0]
    }
}

/// Value reader dispatching between the input matrix and the write buffer via [`At`].
struct Either;

impl Read for Either {
    #[inline]
    fn read<T: Copy>(at: At, values: &[T], written: &[T]) -> T {
        at.of(values, written)
    }
}

/// Metadata for a single eliminated unknown, indexing its pivot, column, row, and updates.
#[derive(Debug, Clone, Copy)]
struct Eliminated {
    /// Column index of the eliminated unknown in the full pattern.
    unknown: usize,
    /// Pivot equation row index in the full pattern.
    row: usize,
    /// Storage address of the pivot entry.
    pivot: At,
    /// Slice bounds in [`Condensation::column`] for this unknown's column entries.
    column: (usize, usize),
    /// Slice bounds in [`Condensation::row`] for this unknown's pivot-row entries.
    entries: (usize, usize),
    /// Starting offset in [`Condensation::updates`] for this step's outer-product updates.
    updates: usize,
    /// Flags `(row_shared, unknown_shared)` indicating whether earlier steps reference this
    /// pivot row or unknown in the solve buffer instead of directly in `right_hand_side`.
    shared: (bool, bool),
    /// Flag indicating that every entry read by this step comes directly from the unmodified
    /// full matrix.
    simple: bool,
}

impl Eliminated {
    /// Returns the index range of this step's column entries in [`Condensation::column`].
    const fn column(&self) -> std::ops::Range<usize> {
        self.column.0..self.column.1
    }

    /// Returns the index range of this step's pivot-row entries in [`Condensation::row`].
    const fn entries(&self) -> std::ops::Range<usize> {
        self.entries.0..self.entries.1
    }
}

/// Numerical state saved during [`Condensation::condense`] for subsequent solves: written
/// Schur-complement values, pivots, column multipliers, and pivot-row entries.
#[derive(Debug, Clone)]
pub(super) struct Saved<T> {
    written: Vec<T>,
    pivots: Vec<T>,
    multipliers: Vec<T>,
    row: Vec<T>,
}

impl<T> Saved<T> {
    /// Creates an empty [`Saved`] buffer with no allocated entries.
    pub(super) const fn new() -> Self {
        Self {
            written: Vec::new(),
            pivots: Vec::new(),
            multipliers: Vec::new(),
            row: Vec::new(),
        }
    }
}

/// Index mapping between full-system rows/columns and reduced retained-system indices after
/// condensation.
struct Numbering {
    /// Retained index for each full unknown, or [`NONE`] when eliminated.
    index: Vec<usize>,
    /// Full unknown index for each retained unknown, in ascending order.
    unknowns: Vec<usize>,
    /// Retained index for each full row (remapped for off-diagonal pivots), or [`NONE`] for
    /// eliminated pivot rows.
    row_index: Vec<usize>,
    /// Full row index corresponding to each retained index.
    rows: Vec<usize>,
    /// Elimination step indices `(row_step, column_step)` for eliminated rows and columns, or
    /// [`NONE`].
    eliminated: (Vec<usize>, Vec<usize>),
}

impl Numbering {
    /// Builds full-to-retained index maps for a matrix of `dimension` unknowns given accepted
    /// elimination steps `taken`.
    fn new(dimension: usize, taken: &[Step]) -> Self {
        let (mut eliminated_row, mut eliminated_column) =
            (vec![NONE; dimension], vec![NONE; dimension]);
        for (index, step) in taken.iter().enumerate() {
            eliminated_row[step.row] = index;
            eliminated_column[step.unknown] = index;
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
        for step in taken.iter().filter(|step| step.row != step.unknown) {
            row_index[step.unknown] = index[step.row];
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

    /// Claims `(row, column)` in `builder` when both the row and column are retained.
    fn claim(&self, builder: &mut PatternBuilder, (row, column): (usize, usize)) -> Option<Entry> {
        let (row, column) = (self.row_index[row], self.index[column]);
        (row != NONE && column != NONE).then(|| builder.entry(row, column))
    }

    /// Returns the solve-buffer index for `row`: its retained index, or `kept + step` if
    /// eliminated.
    fn solved_row(&self, row: usize) -> usize {
        match self.eliminated.0[row] {
            NONE => self.row_index[row],
            elimination => self.unknowns.len() + elimination,
        }
    }

    /// Returns the solve-buffer index for `column`, analogous to [`Numbering::solved_row`].
    fn solved_column(&self, column: usize) -> usize {
        match self.eliminated.1[column] {
            NONE => self.index[column],
            elimination => self.unknowns.len() + elimination,
        }
    }
}

/// Dynamic sparsity tracker combining the original pattern with fill-in added by accepted
/// eliminations.
struct Evolving<'a> {
    starts: &'a [usize],
    rows: &'a [usize],
    full: &'a Pattern,
    filled: HashSet<(usize, usize)>,
    filled_rows: HashMap<usize, Vec<usize>>,
    filled_columns: HashMap<usize, Vec<usize>>,
    /// Flags `(rows_gone, columns_gone)` marking eliminated rows and columns.
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

    /// Looks up the full-pattern slot at `(row, column)`, if stored.
    fn slot(&self, row: usize, column: usize) -> Option<usize> {
        slot_in(self.starts, self.rows, row, column)
    }

    /// Returns whether `(row, column)` exists in the original pattern or accumulated fill-in.
    fn holds(&self, row: usize, column: usize) -> bool {
        self.slot(row, column).is_some() || self.filled.contains(&(row, column))
    }

    /// Returns sorted active row indices in `column`, excluding `pivot` and eliminated rows.
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

    /// Returns sorted active column indices in `row`, excluding `pivot` and eliminated columns.
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

    /// Records a new fill-in entry at `(row, column)`.
    fn fill(&mut self, row: usize, column: usize) {
        self.filled.insert((row, column));
        self.filled_rows.entry(column).or_default().push(row);
        self.filled_columns.entry(row).or_default().push(column);
    }

    /// Eliminates `unknown` using pivot `row`, adding fill-in, clearing `proven` for columns
    /// modified in place, and returning sorted active `(rows, columns)`.
    fn eliminate(
        &mut self,
        (row, unknown): (usize, usize),
        proven: &mut [bool],
    ) -> (Vec<usize>, Vec<usize>) {
        let rows = self.column(unknown, row);
        let columns = self.row(row, unknown);
        for &target in &rows {
            for &along in &columns {
                if self.holds(target, along) {
                    proven[along] = false;
                } else {
                    self.fill(target, along);
                }
            }
        }
        self.gone.0[row] = true;
        self.gone.1[unknown] = true;
        (rows, columns)
    }
}

/// Symbolic layout of every accepted elimination step before slot addresses are resolved.
struct Layout {
    builder: PatternBuilder,
    kept_entries: Vec<(Entry, usize)>,
    later: Vec<usize>,
    list: Vec<Laid>,
    column_entries: Vec<(usize, Place)>,
    row_entries: Vec<(usize, Place)>,
    updates: Vec<Place>,
}

impl Layout {
    /// Records every entry read and written by `taken` against `full` and `numbering`.
    fn build(full: &Pattern, numbering: &Numbering, taken: &[Step]) -> Self {
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
        let structure = full.structure();
        let (col_starts, col_rows) = (structure.col_ptr(), structure.row_idx());
        let (mut list, mut column_entries, mut row_entries, mut updates) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for step in taken {
            let (row, unknown) = (step.row, step.unknown);
            let first = (column_entries.len(), row_entries.len(), updates.len());
            // An entry no elimination before wrote is read from the full matrix.
            let read = |position: (usize, usize)| {
                writes.get(&position).copied().unwrap_or_else(|| {
                    let slot =
                        slot_in(col_starts, col_rows, position.0, position.1).unwrap_or(NONE);
                    debug_assert_ne!(slot, NONE, "unread entry is stored in the full pattern");
                    Place::Full(slot)
                })
            };
            let pivot = read((row, unknown));
            for &each in &step.rows {
                column_entries.push((numbering.solved_row(each), read((each, unknown))));
            }
            for &each in &step.columns {
                row_entries.push((numbering.solved_column(each), read((row, each))));
            }
            for &target in &step.rows {
                for &along in &step.columns {
                    let position = (target, along);
                    let place = *writes.entry(position).or_insert_with(|| {
                        let slot = slot_in(col_starts, col_rows, target, along).unwrap_or(NONE);
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
            list.push(Laid {
                unknown,
                row,
                pivot,
                column: (first.0, column_entries.len()),
                entries: (first.1, row_entries.len()),
                updates: first.2,
            });
        }
        Self {
            builder,
            kept_entries,
            later,
            list,
            column_entries,
            row_entries,
            updates,
        }
    }
}

impl Condensation {
    /// Builds a [`Condensation`] for `full`, accepting every column-dominant `(row, column)`
    /// candidate in `pivots` whose dominance remains proven through elimination.
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
        let mut taken = Vec::new();
        {
            let mut evolving = Evolving::new(&full);
            let mut proven = pivot_column.clone();
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
                let (rows, columns) = evolving.eliminate((row, column), &mut proven);
                taken.push(Step {
                    row,
                    unknown: column,
                    rows,
                    columns,
                });
            }
        }
        Self::from_taken(full, &taken)
    }

    /// Assembles the symbolic [`Condensation`] tables for accepted steps `taken` on `full`.
    fn from_taken(full: Arc<Pattern>, taken: &[Step]) -> Self {
        let numbering = Numbering::new(full.dimension(), taken);
        let Layout {
            builder,
            kept_entries,
            later,
            list,
            column_entries,
            row_entries,
            updates,
        } = Layout::build(&full, &numbering, taken);
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

    /// Returns the original uncondensed matrix pattern.
    #[must_use]
    pub fn full(&self) -> &Arc<Pattern> {
        &self.full
    }

    /// Returns the reduced Schur-complement pattern factored by sparse LU.
    #[must_use]
    pub fn kept(&self) -> &Arc<Pattern> {
        &self.kept
    }

    /// Returns the number of eliminated unknowns.
    #[must_use]
    pub fn eliminated(&self) -> usize {
        self.eliminated.len()
    }

    /// Maps retained unknown `index` back to its full-system unknown index.
    pub(super) fn unknown(&self, index: usize) -> usize {
        self.unknowns[index]
    }

    /// Eliminates all condensed unknowns from `values` into `saved` and returns the slice of
    /// reduced Schur-complement matrix entries.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] naming an eliminated unknown whose pivot is zero
    /// or non-finite.
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
        if pivot.size() == 0.0 || !pivot.is_finite() {
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

    /// Returns the required solve-buffer length for [`Condensation::reduce`] and
    /// [`Condensation::expand`] (retained unknowns plus eliminated unknowns).
    pub(super) fn solved(&self) -> usize {
        self.unknowns.len() + self.eliminated.len()
    }

    /// Reduces `right_hand_side` into `into` (of length [`Condensation::solved`]) for the
    /// kept-unknown system or its transpose.
    ///
    /// Forward elimination updates retained entries and any shared eliminated rows (or shared
    /// eliminated columns when `transposed` is true).
    pub(super) fn reduce<T: Scalar>(
        &self,
        saved: &Saved<T>,
        (right_hand_side, transposed): (&[T], bool),
        into: &mut [T],
    ) {
        let count = self.unknowns.len();
        let kept_indices = if transposed {
            &self.unknowns
        } else {
            &self.rows
        };
        for (into, &index) in into.iter_mut().zip(kept_indices) {
            *into = right_hand_side[index];
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
            into[count + index] = right_hand_side[taken(&self.eliminated[index]).0];
        }
        let ranges = self.eliminated.iter().zip(&saved.pivots).enumerate();
        for (index, (eliminated, &pivot)) in ranges {
            let value = match taken(eliminated) {
                (_, true) => into[count + index],
                (row, false) => right_hand_side[row],
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

    /// Recovers all eliminated unknowns into `right_hand_side` from the kept solution and
    /// reduced buffer in `solved` (in reverse elimination order).
    ///
    /// Evaluates `(b_r − Σ A_rc x_c) / p` for direct solves, or `b_e / p − Σ (A_ke / p) x_k`
    /// when `transposed` is true, reading unshared entries directly from `right_hand_side`.
    pub(super) fn expand<T: Scalar>(
        &self,
        saved: &Saved<T>,
        (solved, transposed): (&mut [T], bool),
        right_hand_side: &mut [T],
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
                right_hand_side[taken]
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
            right_hand_side[recovered] = value;
        }
        let kept_indices = if transposed {
            &self.rows
        } else {
            &self.unknowns
        };
        for (&index, &value) in kept_indices.iter().zip(&solved[..count]) {
            right_hand_side[index] = value;
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

    /// Builds a [`Pattern`] of `dimension` unknowns containing the given `(row, column)` entries.
    fn pattern(dimension: usize, positions: &[(usize, usize)]) -> Arc<Pattern> {
        let mut builder = PatternBuilder::new();
        for &(row, column) in positions {
            builder.entry(row, column);
        }
        Arc::new(builder.finish(dimension))
    }

    #[test]
    fn eliminates_an_unknown_into_the_fill_of_its_neighbors() {
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
        // Eliminating 1 modifies existing entry (0, 2), invalidating column 2's dominance
        // proof so unknown 2 stays kept; unknown 3 couples only to 0 and is eliminated.
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
        // Terminal 0 (conductance 1), inner node 1 (conductance 2), and current 2 through
        // 0.5 ohm between them. Current 2 eliminates via node 1's row (entry −1), and the
        // current row takes node 1's place.
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

    /// Returns a 4x4 strictly diagonally dominant pattern and values where eliminating 1 fills
    /// `(0, 2)` and `(3, 2)` without touching existing entries in column 2, leaving column 2
    /// proven so elimination 2 reads those fills via `Either` (non-simple).
    fn chained_system() -> (Arc<Pattern>, Vec<f64>, usize) {
        let positions = [
            ((0, 0), 4.0),
            ((0, 1), -1.0),
            ((1, 0), -1.0),
            ((1, 1), 4.0),
            ((1, 2), -1.0),
            ((2, 2), 4.0),
            ((2, 3), -1.0),
            ((3, 1), -1.0),
            ((3, 3), 4.0),
        ];
        let mut builder = PatternBuilder::new();
        let entries: Vec<_> = positions
            .iter()
            .map(|&((row, column), _)| builder.entry(row, column))
            .collect();
        let full = Arc::new(builder.finish(4));
        let mut values = vec![0.0; full.len()];
        for (&entry, &(_, value)) in entries.iter().zip(&positions) {
            values[full.slot(entry)] = value;
        }
        let pivot_slot = full.slot(entries[3]);
        (full, values, pivot_slot)
    }

    #[test]
    fn chained_eliminations_match_uncondensed_solves() {
        let (full, real_values, _) = chained_system();
        let condensation = Arc::new(Condensation::new(Arc::clone(&full), &[(1, 1), (2, 2)]));
        assert_eq!(condensation.eliminated(), 2);
        assert!(condensation.eliminated[0].simple);
        assert!(!condensation.eliminated[1].simple);

        let direct_ordering = Arc::new(Ordering::new(full).expect("fits"));
        let condensed_ordering = Arc::new(Ordering::condensed(condensation, None).expect("fits"));

        let mut direct_lu = SparseLu::new(direct_ordering);
        let mut condensed_lu = SparseLu::new(condensed_ordering);
        let right_hand_side = [1.0, -2.0, 3.0, -4.0];
        let (mut direct_x, mut condensed_x) = (right_hand_side, right_hand_side);
        direct_lu
            .factor(&real_values)
            .and_then(|mut f| f.solve(&mut direct_x))
            .expect("direct");
        condensed_lu
            .factor(&real_values)
            .and_then(|mut f| f.solve(&mut condensed_x))
            .expect("condensed");
        for (a, b) in direct_x.iter().zip(&condensed_x) {
            assert!((a - b).abs() < 1e-14, "{direct_x:?} vs {condensed_x:?}");
        }
        let (mut direct_t, mut condensed_t) = (right_hand_side, right_hand_side);
        direct_lu
            .factor(&real_values)
            .and_then(|mut f| f.solve_transpose(&mut direct_t))
            .expect("direct transpose");
        condensed_lu
            .factor(&real_values)
            .and_then(|mut f| f.solve_transpose(&mut condensed_t))
            .expect("condensed transpose");
        for (a, b) in direct_t.iter().zip(&condensed_t) {
            assert!((a - b).abs() < 1e-14, "{direct_t:?} vs {condensed_t:?}");
        }
    }

    #[test]
    fn complex_condensation_matches_uncondensed_and_guards_pivots() {
        let (full, real_values, pivot_slot) = chained_system();
        let condensation = Arc::new(Condensation::new(Arc::clone(&full), &[(1, 1), (2, 2)]));
        let direct_ordering = Arc::new(Ordering::new(full).expect("fits"));
        let condensed_ordering =
            Arc::new(Ordering::condensed(Arc::clone(&condensation), None).expect("fits"));

        let mut complex_values: Vec<num_complex::Complex64> = real_values
            .iter()
            .map(|&value| num_complex::Complex64::new(value, 0.25))
            .collect();
        let complex_rhs = [
            num_complex::Complex64::new(1.0, 0.5),
            num_complex::Complex64::new(-2.0, -0.5),
            num_complex::Complex64::new(3.0, 1.0),
            num_complex::Complex64::new(-4.0, -1.0),
        ];
        let (mut direct_c, mut condensed_c) = (complex_rhs, complex_rhs);
        SparseLu::new(direct_ordering)
            .factor(&complex_values)
            .and_then(|mut f| f.solve(&mut direct_c))
            .expect("direct complex");
        SparseLu::new(Arc::clone(&condensed_ordering))
            .factor(&complex_values)
            .and_then(|mut f| f.solve(&mut condensed_c))
            .expect("condensed complex");
        for (a, b) in direct_c.iter().zip(&condensed_c) {
            assert!((a - b).norm() < 1e-14, "{direct_c:?} vs {condensed_c:?}");
        }
        for bad in [
            num_complex::Complex64::new(0.0, 0.0),
            num_complex::Complex64::new(f64::NAN, 1.0),
            num_complex::Complex64::new(f64::INFINITY, 0.0),
        ] {
            complex_values[pivot_slot] = bad;
            let mut complex_lu = SparseLu::new(Arc::clone(&condensed_ordering));
            assert_eq!(
                complex_lu.factor(&complex_values).map(|_| ()),
                Err(LinearError::Singular { columns: vec![1] })
            );
        }
        complex_values[pivot_slot] = num_complex::Complex64::new(0.0, 1e-310);
        let mut saved = Saved::new();
        assert!(condensation.condense(&complex_values, &mut saved).is_ok());
    }
}
