//! A sparse LU solver for the matrices of circuits, after KLU (Davis and Palamadai Natarajan,
//! "Algorithm 907: KLU, a direct sparse solver for circuit simulation problems", ACM
//! Transactions on Mathematical Software 37(3), 2010).
//!
//! An [`Ordering`] analyzes a pattern once: it matches every column with a row so that the
//! matched entries make a diagonal free of structural zeros, splits the matrix into its
//! block triangular form, and orders each diagonal block to reduce its fill. A [`SparseLu`]
//! then factors the matrices of the pattern, real or complex, by left-looking Gaussian
//! elimination with threshold partial pivoting that prefers the ordering's diagonal, and
//! refactors each new matrix with the pivots of the last, searching afresh only from a pivot
//! that has become too small.
//!
//! An ordering [dissected](Ordering::dissected) orders each large block by nested dissection
//! into interiors and the separators between them, and the factorization of such a block is
//! multifrontal over the tree they make: the interiors factor at once, by the same
//! left-looking elimination, each pivoting within itself, and each separator factors a
//! dense front of the updates the parts below it leave, once they are done. The work
//! spreads over the threads a solver is given, and its factors are the same bits on any
//! number.
//!
//! An ordering of a [`Condensation`] orders the unknowns it keeps: its solvers eliminate the
//! others from each matrix, each by its own pivot, before they factor it, and recover them
//! after each solve, so that their callers give and take the whole system's matrices and
//! vectors.

mod condensation;
mod dissection;
mod numeric;
mod ordering;
mod tree;

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use std::sync::Arc;

use rayon::prelude::*;

use faer::sparse::FaerError;
use num_complex::Complex64;

pub use self::condensation::Condensation;
use self::condensation::Saved;
use self::numeric::{Factorization, Numeric};
pub use self::ordering::{Dissecting, Ordering};
use self::tree::Scratch;
use crate::Pattern;
use crate::reduce::CHUNK;

/// No index: an absent row, column, vertex, pivot, or slot.
///
/// Origin: sentinel `usize::MAX`, unreachable by any valid Rust `Vec` index (allocations are
/// bounded by `isize::MAX`).
const NONE: usize = usize::MAX;

/// A scalar the sparse solver works in: `f64` for DC and transient analyses, [`Complex64`]
/// for small-signal analyses.
pub trait Scalar:
    Copy
    + PartialEq
    + fmt::Debug
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
{
    /// Additive identity (`0`) of the scalar field.
    const ZERO: Self;

    /// The value's magnitude: its absolute value, or a complex number's modulus.
    fn magnitude(self) -> f64;

    /// The value's size in the 1-norm: its absolute value, or a complex number's
    /// `|re| + |im|`, between its modulus and √2 times it, and cheaper, needing no square
    /// root.
    fn size(self) -> f64;

    /// Whether the value is neither infinite nor NaN.
    fn is_finite(self) -> bool;

    /// Whether the value has the same bits as `other`: unlike `==`, a zero is not a
    /// negative zero, and a NaN can be identical to itself.
    fn is_identical(self, other: Self) -> bool;
}

impl Scalar for f64 {
    /// Additive identity of IEEE 754 binary64 reals (`0.0`).
    const ZERO: Self = 0.0;

    fn magnitude(self) -> f64 {
        self.abs()
    }

    fn size(self) -> f64 {
        self.abs()
    }

    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }

    fn is_identical(self, other: Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

impl Scalar for Complex64 {
    /// Additive identity of complex numbers (`0.0 + 0.0i`).
    const ZERO: Self = Self::new(0.0, 0.0);

    fn magnitude(self) -> f64 {
        self.norm()
    }

    fn size(self) -> f64 {
        self.re.abs() + self.im.abs()
    }

    fn is_finite(self) -> bool {
        Complex64::is_finite(self)
    }

    fn is_identical(self, other: Self) -> bool {
        self.re.is_identical(other.re) && self.im.is_identical(other.im)
    }
}

/// Why a linear system could not be solved.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LinearError {
    /// The matrix is singular, structurally or numerically.
    #[error("the system matrix is singular")]
    Singular {
        /// The columns, by index, whose unknowns the system leaves undetermined, as far as
        /// the solver found them, in increasing order: those of a set of columns with
        /// entries in fewer rows than there are columns, for a structurally singular
        /// pattern; the column that has no nonzero pivot left, when elimination finds
        /// one; or else the columns whose values in the solution are not finite.
        columns: Vec<usize>,
    },
    /// The system exceeds the memory or index range available.
    #[error("the system is too large to factor")]
    TooLarge,
}

impl From<FaerError> for LinearError {
    fn from(_: FaerError) -> Self {
        Self::TooLarge
    }
}

impl From<dyn_stack::mem::AllocError> for LinearError {
    fn from(_: dyn_stack::mem::AllocError) -> Self {
        Self::TooLarge
    }
}

impl From<std::collections::TryReserveError> for LinearError {
    fn from(_: std::collections::TryReserveError) -> Self {
        Self::TooLarge
    }
}

/// How much work a [`SparseLu`] has done, by kind, over the factorizations that succeeded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Work {
    /// Factorizations that searched for every pivot: the first, and one after a failure.
    pub factorizations: usize,
    /// Factorizations that kept every pivot of the last, with no search.
    pub refactorizations: usize,
    /// Factorizations that kept the pivots of the last up to one their matrix made too
    /// small, and searched for the pivots from that one on.
    pub fallbacks: usize,
    /// Calls given the very matrix last factored, which kept its factors.
    pub reuses: usize,
}

/// A sparse LU solver for matrices of one [`Pattern`].
///
/// It shares its pattern's [`Ordering`] with any other solver of the pattern, real or
/// complex. Each [`SparseLu::factor`] keeps the factors of the last when given the same
/// matrix, refactors with the last pivots when given another, and searches for pivots only
/// where one has become too small; its workspaces are kept, so the Newton iterations of a
/// whole simulation allocate nothing once the first has run.
///
/// A clone factors every matrix after as the solver it was cloned from would, bit for bit,
/// which lets several threads each go on from one solver's pivots.
///
/// The blocks its ordering [dissects](Ordering::dissected) factor and solve as rayon tasks
/// on the pool they run in when it is [spread](SparseLu::spread) over two threads or more,
/// and on the calling thread alone by default; the factors and the solutions are the same
/// bits either way.
#[derive(Clone)]
pub struct SparseLu<T: Scalar> {
    ordering: Arc<Ordering>,
    /// The threads its work spreads over, as [`SparseLu::spread`] says: one for none.
    threads: usize,
    numeric: Numeric<T>,
    /// The values of the matrix last factored, condensed if its ordering condenses, which
    /// the solves read outside the diagonal blocks.
    factored: Vec<T>,
    /// Whether the factors are those of `factored`.
    current: bool,
    /// What the eliminations made of the last matrix condensed, for an ordering that
    /// condenses, the kept unknowns' matrix first, and what its solves read of the unknowns
    /// eliminated.
    condensed: Saved<T>,
    /// The solves' workspace.
    workspace: Workspace<T>,
    statistics: Work,
}

/// What a solve with [`Shared`] factors writes as it goes: a value for each row of the
/// ordered matrix and for each position. Each thread that solves at once needs its own.
#[derive(Debug, Clone)]
pub struct Workspace<T: Scalar> {
    by_row: Vec<T>,
    by_position: Vec<T>,
    /// Each node's, for the blocks dissected, made at the first solve.
    dissected: Vec<Vec<Scratch<T>>>,
    /// The kept unknowns' right-hand side and solution, for an ordering that condenses,
    /// then what its solves keep of the unknowns eliminated, made at the first solve.
    kept: Vec<T>,
}

impl<T: Scalar> Workspace<T> {
    /// A workspace for the solves of a matrix whose factored pattern, as
    /// [`Ordering::factored`] gives it, has `dimension` rows.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self {
            by_row: vec![T::ZERO; dimension],
            by_position: vec![T::ZERO; dimension],
            dissected: Vec::new(),
            kept: Vec::new(),
        }
    }
}

impl<T: Scalar> SparseLu<T> {
    /// A solver for the matrices of `ordering`'s pattern.
    #[must_use]
    pub fn new(ordering: Arc<Ordering>) -> Self {
        let dimension = ordering.factored().dimension();
        Self {
            numeric: Numeric::new(&ordering),
            threads: 1,
            ordering,
            factored: Vec::new(),
            current: false,
            condensed: Saved::new(),
            workspace: Workspace::new(dimension),
            statistics: Work::default(),
        }
    }

    /// Forgets every factorization, keeping the allocations: the next [`SparseLu::factor`]
    /// searches for every pivot, and the statistics count from zero, as a solver made anew
    /// for the same ordering does, so that it factors every matrix as that one would, bit for
    /// bit.
    pub fn reset(&mut self) {
        self.numeric.forget();
        self.factored.clear();
        self.current = false;
        self.statistics = Work::default();
    }

    /// Factors with `ordering`, of the same pattern, from now on, forgetting every
    /// factorization as [`SparseLu::reset`] does but keeping its counts of work: it factors
    /// every matrix as a solver made anew for `ordering` would, bit for bit.
    ///
    /// # Panics
    ///
    /// Panics if `ordering` is of another pattern.
    pub fn reorder(&mut self, ordering: Arc<Ordering>) {
        assert!(
            Arc::ptr_eq(ordering.pattern(), self.ordering.pattern()),
            "an ordering of the same pattern"
        );
        let (statistics, threads) = (self.statistics, self.threads);
        *self = Self::new(ordering);
        self.statistics = statistics;
        self.threads = threads;
    }

    /// Takes on `other`'s ordering, pivots and factors, keeping its own counts of work and
    /// the threads it spreads over, as [`SparseLu::reorder`] keeps them: from now on it
    /// factors every matrix as `other` would, bit for bit, and counts the work as its own.
    pub fn adopt(&mut self, other: &Self) {
        let (statistics, threads) = (self.statistics, self.threads);
        self.clone_from(other);
        self.statistics = statistics;
        self.threads = threads;
    }

    /// Spreads its work over the rayon pool it runs in, for two `threads` or more: the
    /// factorizations and solves of the blocks its ordering dissects as rayon tasks, one for
    /// each subtree of a block's dissection down to its interiors, which the pool's threads
    /// take as they free, each separator's front spreading its columns over its share of
    /// `threads`, in proportion to its subtree's positions; and the solves' permutations,
    /// the copy of the values factored and the check of the solutions in chunks of
    /// [`crate::reduce::CHUNK`]. A pool of `threads` threads keeps the work to as many at
    /// once. For one, everything runs on the calling thread alone. The factors and
    /// solutions are the same bits either way.
    pub fn spread(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// The threads its work spreads over, as [`SparseLu::spread`] says: one when it
    /// spreads none.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The ordering of the pattern this solver factors.
    #[must_use]
    pub fn ordering(&self) -> &Arc<Ordering> {
        &self.ordering
    }

    /// The pattern of the matrices this solver factors.
    #[must_use]
    pub fn pattern(&self) -> &Arc<Pattern> {
        self.ordering.pattern()
    }

    /// How much work the solver has done.
    #[must_use]
    pub fn statistics(&self) -> Work {
        self.statistics
    }

    /// Whether this solver factors every matrix from now on as `other` would, bit for bit:
    /// both share one ordering and keep the same pivots, and so the same patterns of `L`
    /// and `U`, or neither keeps any. A factorization depends on nothing else the solvers
    /// hold: the factors a solver keeps for the matrix it factored last are those a
    /// refactorization of it with the same pivots makes. Only their counts of work may
    /// differ. Solvers of two orderings are never alike, even of one pattern.
    #[must_use]
    pub fn factors_alike(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.ordering, &other.ordering) && self.numeric.pivots_alike(&other.numeric)
    }

    /// The entries the solves read: those of `L` below its unit diagonal, those of `U` on
    /// and above its diagonal, and the matrix's own entries above its diagonal blocks, as
    /// the last factorization left them. The fill is their number less the matrix's.
    #[must_use]
    pub fn entries(&self) -> usize {
        self.numeric.entries() + self.ordering.entries_above()
    }

    /// The number of columns the last factorization delayed, in the blocks its ordering
    /// [dissects](Ordering::dissected): each a column whose best pivot among the rows of its
    /// interior or separator fell short of a tenth of the largest entry below it, moved
    /// with a row its node did not pivot to the separator above, which pivoted it among
    /// more rows.
    #[must_use]
    pub fn delays(&self) -> usize {
        self.numeric.delays()
    }

    /// Factors the matrix whose stored values, in the pattern's order, are `values`.
    ///
    /// The factors of the last matrix are kept when every value is bitwise identical to its
    /// own. Otherwise the pivots of the last factorization are kept while each stays at
    /// least a tenth of the largest entry below it in its column, and the pivots are
    /// searched for from the first that does not, each the diagonal entry of the ordering
    /// where it is that large, and else the largest.
    ///
    /// An ordering that condenses has the matrix condensed first, and the factors are those
    /// of its kept unknowns' matrix, kept when that is bitwise identical to the last.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] for every matrix of a structurally singular
    /// pattern, naming its [undetermined](Ordering::undetermined) columns, and for a matrix
    /// with a column that has no nonzero pivot left once the columns before it are
    /// eliminated, or an unknown condensed whose pivot is zero or not finite, naming that
    /// column; and [`LinearError::TooLarge`] when the factors do not fit in memory.
    ///
    /// # Panics
    ///
    /// Panics if `values` does not hold one value per stored position of the pattern.
    pub fn factor(&mut self, values: &[T]) -> Result<Factors<'_, T>, LinearError> {
        assert_eq!(
            values.len(),
            self.ordering.pattern().len(),
            "one value per stored position"
        );
        let values = match self.ordering.condensation() {
            Some(condensation) => match condensation.condense(values, &mut self.condensed) {
                Ok(kept) => kept,
                Err(error) => {
                    self.current = false;
                    return Err(error);
                }
            },
            None => values,
        };
        let unchanged = self.factored.len() == values.len()
            && self
                .factored
                .iter()
                .zip(values)
                .all(|(&factored, &value)| factored.is_identical(value));
        if self.current && unchanged {
            self.statistics.reuses += 1;
        } else {
            self.current = false;
            let undetermined = self.ordering.undetermined();
            if !undetermined.is_empty() {
                return Err(LinearError::Singular {
                    columns: undetermined.to_vec(),
                });
            }
            copy(&mut self.factored, values, self.threads);
            let factored = self.numeric.factor(&self.ordering, values, self.threads);
            let counted = match factored.map_err(|error| kept_to_full(&self.ordering, error))? {
                Factorization::Searched => &mut self.statistics.factorizations,
                Factorization::Refactored => &mut self.statistics.refactorizations,
                Factorization::FellBack => &mut self.statistics.fallbacks,
            };
            *counted += 1;
            self.current = true;
        }
        Ok(Factors {
            shared: Shared {
                ordering: &self.ordering,
                numeric: &self.numeric,
                values: &self.factored,
                saved: &self.condensed,
                threads: self.threads,
            },
            workspace: &mut self.workspace,
        })
    }
}

/// `error`, of the factorization of `ordering`'s factored pattern, naming the columns of its
/// whole pattern: those of the unknowns its condensation keeps, if it condenses.
fn kept_to_full(ordering: &Ordering, error: LinearError) -> LinearError {
    match (ordering.condensation(), error) {
        (Some(condensation), LinearError::Singular { columns }) => LinearError::Singular {
            columns: columns
                .into_iter()
                .map(|column| condensation.unknown(column))
                .collect(),
        },
        (_, error) => error,
    }
}

impl<T: Scalar> fmt::Debug for SparseLu<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseLu")
            .field("ordering", &self.ordering)
            .field("statistics", &self.statistics)
            .finish_non_exhaustive()
    }
}

/// The LU factors of one matrix, ready to solve systems with it.
pub struct Factors<'a, T: Scalar> {
    shared: Shared<'a, T>,
    workspace: &'a mut Workspace<T>,
}

impl<T: Scalar> fmt::Debug for Factors<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.shared.fmt(f)
    }
}

impl<'a, T: Scalar> Factors<'a, T> {
    /// Solves `A x = right_hand_side` and leaves `x` in `right_hand_side`.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] when the solution is not finite, which a pivot
    /// too small for the matrix's range causes, naming the columns whose values are not.
    ///
    /// # Panics
    ///
    /// Panics if `right_hand_side` does not hold one value per row.
    pub fn solve(&mut self, right_hand_side: &mut [T]) -> Result<(), LinearError> {
        self.shared.solve(right_hand_side, self.workspace)
    }

    /// Solves `Aᵀ x = right_hand_side`, the transposed system, and leaves `x` in
    /// `right_hand_side`: the adjoint system whose solution weighs every equation's effect on
    /// one unknown.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] when the solution is not finite, which a pivot
    /// too small for the matrix's range causes, naming the columns whose values are not.
    ///
    /// # Panics
    ///
    /// Panics if `right_hand_side` does not hold one value per row.
    pub fn solve_transpose(&mut self, right_hand_side: &mut [T]) -> Result<(), LinearError> {
        self.shared.solve_transpose(right_hand_side, self.workspace)
    }

    /// The factors, to solve with from several threads at once, each with a [`Workspace`]
    /// of its own: each solve gives the bits [`Factors::solve`] would.
    #[must_use]
    pub fn shared(&self) -> Shared<'a, T> {
        self.shared
    }
}

/// The LU factors of one matrix, which any number of threads solve systems with at once,
/// each writing to a [`Workspace`] of its own.
#[derive(Clone, Copy)]
pub struct Shared<'a, T: Scalar> {
    ordering: &'a Ordering,
    numeric: &'a Numeric<T>,
    values: &'a [T],
    /// What the solves read of the unknowns a condensing ordering eliminates.
    saved: &'a Saved<T>,
    /// The threads a solve spreads over, as [`SparseLu::spread`] says.
    threads: usize,
}

impl<T: Scalar> fmt::Debug for Shared<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Factors")
            .field("rows", &self.ordering.pattern().dimension())
            .finish_non_exhaustive()
    }
}

impl<T: Scalar> Shared<'_, T> {
    /// A workspace for its solves, as [`Workspace::new`] makes one for its factored
    /// pattern.
    #[must_use]
    pub fn workspace(&self) -> Workspace<T> {
        Workspace::new(self.ordering.factored().dimension())
    }

    /// Solves `A x = right_hand_side` and leaves `x` in `right_hand_side`, writing to
    /// `workspace` as it goes.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] when the solution is not finite, which a pivot
    /// too small for the matrix's range causes, naming the columns whose values are not.
    ///
    /// # Panics
    ///
    /// Panics if `right_hand_side` or `workspace` does not hold one value per row.
    pub fn solve(
        &self,
        right_hand_side: &mut [T],
        workspace: &mut Workspace<T>,
    ) -> Result<(), LinearError> {
        self.solve_either(right_hand_side, workspace, false)
    }

    /// Solves `Aᵀ x = right_hand_side` and leaves `x` in `right_hand_side`, writing to
    /// `workspace` as it goes.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] when the solution is not finite, which a pivot
    /// too small for the matrix's range causes, naming the columns whose values are not.
    ///
    /// # Panics
    ///
    /// Panics if `right_hand_side` or `workspace` does not hold one value per row.
    pub fn solve_transpose(
        &self,
        right_hand_side: &mut [T],
        workspace: &mut Workspace<T>,
    ) -> Result<(), LinearError> {
        self.solve_either(right_hand_side, workspace, true)
    }

    /// Solves `A x = right_hand_side`, or `Aᵀ x = right_hand_side` when `transposed`, and
    /// leaves `x` in `right_hand_side`: for an ordering that condenses, the kept unknowns'
    /// system of `right_hand_side` first, then each unknown eliminated from it.
    fn solve_either(
        &self,
        right_hand_side: &mut [T],
        workspace: &mut Workspace<T>,
        transposed: bool,
    ) -> Result<(), LinearError> {
        assert_eq!(
            right_hand_side.len(),
            self.ordering.pattern().dimension(),
            "one right-hand side value per row"
        );
        let dimension = self.ordering.factored().dimension();
        assert_eq!(
            workspace.by_row.len(),
            dimension,
            "a workspace of one value per factored row"
        );
        let Some(condensation) = self.ordering.condensation() else {
            self.solve_factored(right_hand_side, workspace, transposed);
            return finite_on(right_hand_side, self.threads);
        };
        let mut kept = std::mem::take(&mut workspace.kept);
        kept.resize(condensation.solved(), T::ZERO);
        condensation.reduce(self.saved, (&*right_hand_side, transposed), &mut kept);
        self.solve_factored(&mut kept[..dimension], workspace, transposed);
        condensation.expand(self.saved, (&mut kept, transposed), right_hand_side);
        workspace.kept = kept;
        finite_on(right_hand_side, self.threads)
    }

    /// Solves the factored pattern's system of `right_hand_side`, transposed when
    /// `transposed`.
    fn solve_factored(
        &self,
        right_hand_side: &mut [T],
        workspace: &mut Workspace<T>,
        transposed: bool,
    ) {
        let factors = (self.ordering, self.values);
        if transposed {
            self.numeric
                .solve_transpose(factors, right_hand_side, workspace, self.threads);
        } else {
            self.numeric
                .solve(factors, right_hand_side, workspace, self.threads);
        }
    }
}

/// Sets `into` to the values of `from`, in chunks of [`CHUNK`] spread over the current rayon
/// pool's threads when `threads` is two or more, `from` spans more than one chunk, and `into`
/// holds as many already.
fn copy<T: Scalar>(into: &mut Vec<T>, from: &[T], threads: usize) {
    if threads < 2 || from.len() <= CHUNK || into.len() != from.len() {
        if into.len() == from.len() {
            into.copy_from_slice(from);
        } else {
            into.clear();
            into.extend_from_slice(from);
        }
        return;
    }
    into.par_chunks_mut(CHUNK)
        .zip(from.par_chunks(CHUNK))
        .for_each(|(into, from)| into.copy_from_slice(from));
}

/// [`finite`], whose check of the values runs in chunks of [`CHUNK`] spread over the
/// current rayon pool's threads when `threads` is two or more and `solution` spans more than
/// one chunk.
fn finite_on<T: Scalar>(solution: &[T], threads: usize) -> Result<(), LinearError> {
    let all = |chunk: &[T]| chunk.iter().all(|&value| value.is_finite());
    if threads >= 2 && solution.len() > CHUNK && solution.par_chunks(CHUNK).all(all) {
        return Ok(());
    }
    finite(solution)
}

/// `Ok` when every value of a solution is finite, and else the singular error that names
/// the columns whose values are not.
pub(crate) fn finite<T: Scalar>(solution: &[T]) -> Result<(), LinearError> {
    if solution.iter().all(|&value| value.is_finite()) {
        return Ok(());
    }
    let columns = solution
        .iter()
        .enumerate()
        .filter(|&(_, &value)| !value.is_finite())
        .map(|(column, _)| column)
        .collect();
    Err(LinearError::Singular { columns })
}
