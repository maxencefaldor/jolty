//! The static condensation of a pattern's matrices: some of their unknowns eliminated, each
//! by its own diagonal pivot, before the sparse factorization of the rest, and recovered
//! after each solve.

use std::fmt;
use std::sync::Arc;

use super::{LinearError, Scalar};
use crate::{Entry, Pattern, PatternBuilder};

/// No index: an unknown eliminated, a value only an elimination fills.
const NONE: usize = usize::MAX;

/// Which unknowns of a pattern's matrices are eliminated before the sparse factorization,
/// and the pattern of the unknowns kept.
///
/// Eliminating the unknown `e` of `A = [[A_kk, A_ke], [A_ek, A_ee]]` by its pivot
/// `p = A_ee` leaves the kept unknowns the matrix `A_kk − (A_ke/p) A_ek`, a Schur
/// complement, which the sparse factorization factors; a system `A x = b` is then solved as
/// the kept unknowns' system with the right-hand side `b_k − (A_ke/p) b_e`, and
/// `x_e = (b_e − A_ek x_k)/p` recovers the unknown eliminated. In exact arithmetic that is
/// the solution of the whole system: the static condensation of finite elements (Wilson,
/// "The static condensation algorithm", International Journal for Numerical Methods in
/// Engineering 8(1), 1974).
///
/// The unknowns given are those whose pivots their matrices' makers prove to be at least
/// every other entry of their columns, so that each passes the factorization's threshold
/// without a search. Of them, one is eliminated only if its pivot is stored and no other of
/// them shares its row or its column: each pivot's column is then the one its proof is
/// about when it is eliminated, whatever was eliminated before it. The others stay among
/// the unknowns kept.
///
/// Every sum runs in one order: the unknowns eliminated in increasing order, each one's
/// column and row in increasing order of the kept unknowns.
pub struct Condensation {
    /// The pattern of the matrices condensed.
    full: Arc<Pattern>,
    /// The pattern of the kept unknowns' matrix: the full pattern's entries among them, and
    /// those the eliminations fill.
    kept: Arc<Pattern>,
    /// The full pattern's unknown of each kept unknown, in increasing order.
    unknowns: Vec<usize>,
    /// The full pattern's slot of each slot of the kept pattern, or [`NONE`] for one only an
    /// elimination fills.
    sources: Vec<usize>,
    /// Each unknown eliminated, in increasing order.
    eliminated: Vec<Eliminated>,
    /// The kept row and the full pattern's slot of each entry of the eliminated unknowns'
    /// columns but their pivots, unknown after unknown, each one's in increasing row.
    column: Vec<(usize, usize)>,
    /// The kept column and the full pattern's slot of each entry of their rows but their
    /// pivots, unknown after unknown, each one's in increasing column.
    row: Vec<(usize, usize)>,
    /// The kept pattern's slot each product of an elimination updates: unknown after
    /// unknown, for each entry of its column, for each entry of its row.
    updates: Vec<usize>,
}

/// One unknown eliminated: where its pivot, its column, its row and its updates lie.
#[derive(Debug, Clone, Copy)]
struct Eliminated {
    /// The unknown, in the full pattern.
    unknown: usize,
    /// The full pattern's slot of its pivot.
    pivot: usize,
    /// Where its column's entries lie among [`Condensation::column`].
    column: (usize, usize),
    /// Where its row's entries lie among [`Condensation::row`].
    row: (usize, usize),
    /// Where its updates begin among [`Condensation::updates`].
    updates: usize,
}

impl Eliminated {
    /// The range of its column's entries.
    const fn column(&self) -> std::ops::Range<usize> {
        self.column.0..self.column.1
    }

    /// The range of its row's entries.
    const fn row(&self) -> std::ops::Range<usize> {
        self.row.0..self.row.1
    }
}

/// What a factorization of a condensed matrix keeps for its solves: each eliminated
/// unknown's pivot, the multipliers of its column, each entry among the kept unknowns over
/// the pivot, and the entries of its row among them, in the order [`Condensation`] lists
/// them.
#[derive(Debug, Clone)]
pub(super) struct Saved<T> {
    pivots: Vec<T>,
    multipliers: Vec<T>,
    row: Vec<T>,
}

impl<T> Saved<T> {
    /// Nothing saved yet.
    pub(super) const fn new() -> Self {
        Self {
            pivots: Vec::new(),
            multipliers: Vec::new(),
            row: Vec::new(),
        }
    }
}

impl Condensation {
    /// The condensation of `full`'s matrices that eliminates those of the unknowns
    /// `pivoted` that it can: each whose pivot is stored and whose row and column hold no
    /// other of them. The pivot of each is to be at least every other entry of its column
    /// in every matrix condensed.
    #[must_use]
    pub fn new(full: Arc<Pattern>, pivoted: &[usize]) -> Self {
        let dimension = full.dimension();
        let mut candidate = vec![false; dimension];
        for &unknown in pivoted {
            if let Some(mark) = candidate.get_mut(unknown) {
                *mark = true;
            }
        }
        let structure = full.structure();
        let (starts, rows) = (structure.col_ptr(), structure.row_idx());
        let column_of =
            |unknown: usize| (starts[unknown]..starts[unknown + 1]).map(|slot| (rows[slot], slot));
        // An unknown is eliminated when its pivot is stored and no other candidate shares
        // its row or its column.
        let eliminated: Vec<bool> = (0..dimension)
            .map(|unknown| {
                candidate[unknown]
                    && column_of(unknown).any(|(row, _)| row == unknown)
                    && column_of(unknown).all(|(row, _)| row == unknown || !candidate[row])
                    && full
                        .row(unknown)
                        .iter()
                        .all(|&(_, column)| column == unknown || !candidate[column])
            })
            .collect();
        let mut index = vec![NONE; dimension];
        let mut unknowns = Vec::new();
        for (unknown, _) in eliminated.iter().enumerate().filter(|&(_, &out)| !out) {
            index[unknown] = unknowns.len();
            unknowns.push(unknown);
        }
        let mut builder = PatternBuilder::new();
        // The full pattern's entries among the kept unknowns, in its order of slots.
        let mut copied = Vec::new();
        for (slot, (row, column)) in full.positions().enumerate() {
            if index[row] != NONE && index[column] != NONE {
                copied.push((builder.entry(index[row], index[column]), slot));
            }
        }
        let (mut list, mut column, mut row, mut filled) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for unknown in (0..dimension).filter(|&unknown| eliminated[unknown]) {
            let first = (column.len(), row.len());
            let mut pivot = NONE;
            for (each, slot) in column_of(unknown) {
                if each == unknown {
                    pivot = slot;
                } else {
                    column.push((index[each], slot));
                }
            }
            row.extend(
                full.row(unknown)
                    .iter()
                    .filter(|&&(_, each)| each != unknown)
                    .map(|&(slot, each)| (index[each], slot)),
            );
            let updates = filled.len();
            for &(target, _) in &column[first.0..] {
                for &(along, _) in &row[first.1..] {
                    filled.push(builder.entry(target, along));
                }
            }
            list.push(Eliminated {
                unknown,
                pivot,
                column: (first.0, column.len()),
                row: (first.1, row.len()),
                updates,
            });
        }
        let kept = builder.finish(unknowns.len());
        let mut sources = vec![NONE; kept.len()];
        for (entry, slot) in copied {
            sources[kept.slot(entry)] = slot;
        }
        let updates = filled
            .into_iter()
            .map(|entry: Entry| kept.slot(entry))
            .collect();
        Self {
            full,
            kept: Arc::new(kept),
            unknowns,
            sources,
            eliminated: list,
            column,
            row,
            updates,
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

    /// Writes into `kept` the kept unknowns' matrix of the full matrix whose stored values
    /// are `values`, and into `saved` what its solves need.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] naming an eliminated unknown whose pivot is zero
    /// or not finite.
    pub(super) fn condense<T: Scalar>(
        &self,
        values: &[T],
        kept: &mut Vec<T>,
        saved: &mut Saved<T>,
    ) -> Result<(), LinearError> {
        kept.clear();
        kept.extend(
            self.sources
                .iter()
                .map(|&slot| if slot == NONE { T::ZERO } else { values[slot] }),
        );
        let Saved {
            pivots,
            multipliers,
            row,
        } = saved;
        pivots.clear();
        multipliers.clear();
        row.clear();
        row.extend(self.row.iter().map(|&(_, slot)| values[slot]));
        for eliminated in &self.eliminated {
            let pivot = values[eliminated.pivot];
            if pivot.magnitude() == 0.0 || !pivot.is_finite() {
                return Err(LinearError::Singular {
                    columns: vec![eliminated.unknown],
                });
            }
            pivots.push(pivot);
            let mut updates = self.updates[eliminated.updates..].iter();
            for &(_, slot) in &self.column[eliminated.column()] {
                let multiplier = values[slot] / pivot;
                multipliers.push(multiplier);
                for (&along, &slot) in row[eliminated.row()].iter().zip(&mut updates) {
                    kept[slot] -= multiplier * along;
                }
            }
        }
        Ok(())
    }

    /// Writes into `into`, one value per kept unknown, the right-hand side of the kept
    /// unknowns' system of the full system whose right-hand side is `rhs`, of the matrix
    /// `saved` was made of: `b_k − (A_ke/p) b_e`, or `b_k − (A_ek/p)ᵀ b_e` when
    /// `transposed`.
    pub(super) fn reduce<T: Scalar>(
        &self,
        saved: &Saved<T>,
        (rhs, transposed): (&[T], bool),
        into: &mut [T],
    ) {
        for (into, &unknown) in into.iter_mut().zip(&self.unknowns) {
            *into = rhs[unknown];
        }
        for (eliminated, &pivot) in self.eliminated.iter().zip(&saved.pivots) {
            let value = rhs[eliminated.unknown];
            if transposed {
                let (entries, row) = (&self.row[eliminated.row()], &saved.row[eliminated.row()]);
                for (&(kept, _), &entry) in entries.iter().zip(row) {
                    into[kept] -= entry / pivot * value;
                }
            } else {
                let range = eliminated.column();
                let (entries, multipliers) =
                    (&self.column[range.clone()], &saved.multipliers[range]);
                for (&(kept, _), &multiplier) in entries.iter().zip(multipliers) {
                    into[kept] -= multiplier * value;
                }
            }
        }
    }

    /// Writes into `rhs`, the right-hand side `reduce` was given, the solution of the full
    /// system from `solved`, the kept unknowns' solution, and each eliminated unknown
    /// recovered from its own right-hand side: `(b_e − A_ek x_k)/p`, or
    /// `b_e/p − (A_ke/p)ᵀ x_k` when `transposed`.
    pub(super) fn expand<T: Scalar>(
        &self,
        saved: &Saved<T>,
        (solved, transposed): (&[T], bool),
        rhs: &mut [T],
    ) {
        for (eliminated, &pivot) in self.eliminated.iter().zip(&saved.pivots) {
            let value = rhs[eliminated.unknown];
            rhs[eliminated.unknown] = if transposed {
                let range = eliminated.column();
                let (entries, multipliers) =
                    (&self.column[range.clone()], &saved.multipliers[range]);
                entries
                    .iter()
                    .zip(multipliers)
                    .fold(value / pivot, |value, (&(kept, _), &multiplier)| {
                        value - multiplier * solved[kept]
                    })
            } else {
                let (entries, row) = (&self.row[eliminated.row()], &saved.row[eliminated.row()]);
                entries
                    .iter()
                    .zip(row)
                    .fold(value, |value, (&(kept, _), &entry)| {
                        value - entry * solved[kept]
                    })
                    / pivot
            };
        }
        for (&unknown, &value) in self.unknowns.iter().zip(solved) {
            rhs[unknown] = value;
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
        let condensation = Condensation::new(full, &[2]);
        assert_eq!(condensation.eliminated(), 1);
        let kept: Vec<_> = condensation.kept().positions().collect();
        assert_eq!(kept, [(0, 0), (0, 1), (1, 1)]);
    }

    #[test]
    fn keeps_the_unknowns_that_share_a_row_or_a_column() {
        // 1 and 2 share an entry, so neither is eliminated; 3 is, alone in its row and
        // column but for 0.
        let full = pattern(
            4,
            &[
                (0, 0),
                (1, 1),
                (2, 2),
                (3, 3),
                (1, 2),
                (0, 1),
                (2, 0),
                (0, 3),
                (3, 0),
            ],
        );
        let condensation = Condensation::new(full, &[1, 2, 3]);
        assert_eq!(condensation.eliminated(), 1);
        assert_eq!(condensation.kept().dimension(), 3);
        // An unknown without a stored pivot stays too.
        let unpivoted = Condensation::new(pattern(2, &[(0, 0), (0, 1), (1, 0)]), &[1]);
        assert_eq!(unpivoted.eliminated(), 0);
    }

    #[test]
    fn names_an_eliminated_unknown_whose_pivot_is_zero() {
        let full = pattern(2, &[(0, 0), (1, 1), (0, 1), (1, 0)]);
        let condensation = Arc::new(Condensation::new(Arc::clone(&full), &[1]));
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
