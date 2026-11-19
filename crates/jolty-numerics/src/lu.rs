//! Sparse LU solver for circuit matrices modeled on KLU (Davis and Palamadai Natarajan,
//! "Algorithm 907: KLU, a direct sparse solver for circuit simulation problems", ACM
//! Transactions on Mathematical Software 37(3), 2010).
//!
//! [`Ordering`] analyzes a sparsity pattern once: matching columns to rows for a zero-free
//! diagonal, partitioning into block triangular form, and ordering each diagonal block to
//! limit fill. [`SparseLu`] then factors real or complex matrices using left-looking Gaussian
//! elimination with threshold partial pivoting and pivot reuse across refactorizations.
//!
//! A [dissected](Ordering::dissected) ordering splits large blocks via nested dissection into
//! interiors and separators for multifrontal parallel factorization. Interiors factor
//! concurrently with local pivoting, and separators factor dense fronts from child updates,
//! producing bit-identical results across any thread count.
//!
//! An ordering built with [`Condensation`] orders only the kept unknowns; its solvers
//! eliminate the remaining unknowns prior to factorization and recover them after each solve
//! so callers still pass full-system matrices and vectors.

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

/// Sentinel index representing an absent row, column, vertex, pivot, or slot.
const NONE: usize = usize::MAX;

/// Scalar field supported by the sparse solver: `f64` for DC and transient analyses, and
/// [`Complex64`] for small-signal frequency-domain analyses.
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
    /// Zero element of the scalar field.
    const ZERO: Self;

    /// Returns the scalar magnitude (absolute value for reals, modulus for complex values).
    fn magnitude(self) -> f64;

    /// Returns the 1-norm size (`|x|` for reals, `|re| + |im|` for complex values), avoiding a
    /// square root while staying within `sqrt(2)` of the modulus.
    fn size(self) -> f64;

    /// Returns whether the scalar is finite (neither infinite nor NaN).
    fn is_finite(self) -> bool;

    /// Returns whether `self` and `other` have identical bit patterns (distinguishing `-0.0`
    /// from `0.0` and treating identical NaN payloads as equal).
    fn is_identical(self, other: Self) -> bool;
}

impl Scalar for f64 {
    /// Real zero (`0.0`).
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
    /// Complex zero (`0.0 + 0.0i`).
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

/// Error returned when a linear system cannot be factored or solved.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LinearError {
    /// Structurally or numerically singular matrix.
    #[error("the system matrix is singular")]
    Singular {
        /// Sorted column indices left undetermined by the solver: structurally rank-deficient
        /// columns, the column lacking a nonzero pivot during elimination, or columns with
        /// non-finite solution values.
        columns: Vec<usize>,
    },
    /// Matrix or factor storage exceeds available memory or index limits.
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

/// Cumulative factorization statistics recorded by a [`SparseLu`] solver.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Work {
    /// Full factorizations that searched for every pivot (initial or post-failure).
    pub factorizations: usize,
    /// Refactorizations that reused every pivot from the prior factorization without search.
    pub refactorizations: usize,
    /// Partial refactorizations that fell back to pivot search after a reused pivot dropped
    /// below threshold.
    pub fallbacks: usize,
    /// Factor calls given a bitwise-identical matrix that reused existing factors unchanged.
    pub reuses: usize,
}

/// Sparse LU solver for square matrices sharing one [`Pattern`].
///
/// A solver shares its pattern's [`Ordering`] with any other real or complex solver for the
/// same pattern. [`SparseLu::factor`] reuses factors on bitwise-identical matrices, reuses
/// prior pivots when acceptable, and retains workspaces across Newton iterations without extra
/// allocations.
///
/// Cloning preserves pivot state so multiple threads can factor subsequent matrices bit for
/// bit identically from the same baseline. When [`SparseLu::spread`] configures two or more
/// threads, [dissected](Ordering::dissected) blocks factor and solve across rayon tasks with
/// bit-identical results.
#[derive(Clone)]
pub struct SparseLu<T: Scalar> {
    ordering: Arc<Ordering>,
    /// Target thread count configured by [`SparseLu::spread`] (`1` runs single-threaded).
    threads: usize,
    numeric: Numeric<T>,
    /// Values of the last factored (and optionally condensed) matrix read outside diagonal
    /// blocks during solves.
    factored: Vec<T>,
    /// Flag indicating that `numeric` holds valid factors for `factored`.
    current: bool,
    /// Condensed matrix values and elimination multipliers saved for condensing orderings.
    condensed: Saved<T>,
    /// Reusable solve workspace owned by the solver.
    workspace: Workspace<T>,
    statistics: Work,
}

/// Per-thread scratch storage for solving with [`Shared`] LU factors.
///
/// Each concurrent thread solving with shared factors requires a separate workspace instance.
#[derive(Debug, Clone)]
pub struct Workspace<T: Scalar> {
    by_row: Vec<T>,
    by_position: Vec<T>,
    /// Per-node scratch buffers for dissected blocks, allocated lazily on first solve.
    dissected: Vec<Vec<Scratch<T>>>,
    /// Reduced right-hand side and elimination scratch for condensing orderings.
    kept: Vec<T>,
}

impl<T: Scalar> Workspace<T> {
    /// Allocates a solve workspace for a factored pattern of `dimension` rows
    /// ([`Ordering::factored`]).
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
    /// Creates a sparse LU solver for matrices matching `ordering`.
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

    /// Clears cached factors, pivots, and work statistics while retaining workspace memory.
    ///
    /// Subsequent [`SparseLu::factor`] calls behave bit for bit identically to a freshly
    /// constructed solver with the same ordering.
    pub fn reset(&mut self) {
        self.numeric.forget();
        self.factored.clear();
        self.current = false;
        self.statistics = Work::default();
    }

    /// Switches to `ordering` for the same pattern, clearing cached factors while preserving
    /// work statistics and thread count.
    ///
    /// # Panics
    ///
    /// Panics if `ordering` belongs to a different pattern.
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

    /// Adopts the ordering, pivots, and cached factors of `other` while keeping this solver's
    /// work statistics and thread count.
    pub fn adopt(&mut self, other: &Self) {
        let (statistics, threads) = (self.statistics, self.threads);
        self.clone_from(other);
        self.statistics = statistics;
        self.threads = threads;
    }

    /// Configures parallel execution over `threads` rayon worker threads (`1` runs solely on
    /// the calling thread).
    ///
    /// For two or more threads, dissected subtrees execute as rayon tasks, separator fronts
    /// distribute columns proportionally across threads, and permutations, copies, and
    /// finiteness checks run in [`crate::reduce::CHUNK`] slices. Factors and solutions are
    /// bit-identical for any thread count.
    pub fn spread(&mut self, threads: usize) {
        self.threads = threads.max(1);
    }

    /// Returns the thread count configured via [`SparseLu::spread`] (`1` when sequential).
    #[must_use]
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Returns the shared [`Ordering`] used by this solver.
    #[must_use]
    pub fn ordering(&self) -> &Arc<Ordering> {
        &self.ordering
    }

    /// Returns the underlying matrix [`Pattern`] used by this solver.
    #[must_use]
    pub fn pattern(&self) -> &Arc<Pattern> {
        self.ordering.pattern()
    }

    /// Returns cumulative factorization statistics recorded by this solver.
    #[must_use]
    pub fn statistics(&self) -> Work {
        self.statistics
    }

    /// Returns whether this solver and `other` share the same [`Ordering`] and identical pivot
    /// state, guaranteeing bit-identical future factorizations.
    ///
    /// Work statistics may differ, and solvers using distinct [`Ordering`] instances always
    /// return `false`.
    #[must_use]
    pub fn factors_alike(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.ordering, &other.ordering) && self.numeric.pivots_alike(&other.numeric)
    }

    /// Returns the total nonzero count read during solves: subdiagonal `L`, upper-triangular
    /// `U`, and off-diagonal block entries from the matrix.
    ///
    /// Fill-in equals this count minus the original matrix entry count.
    #[must_use]
    pub fn entries(&self) -> usize {
        self.numeric.entries() + self.ordering.entries_above()
    }

    /// Returns the number of delayed pivot columns in the last factorization across
    /// [dissected](Ordering::dissected) blocks.
    ///
    /// A column is delayed to its parent separator when its best local pivot falls below
    /// one-tenth of the largest column entry.
    #[must_use]
    pub fn delays(&self) -> usize {
        self.numeric.delays()
    }

    /// Factors the matrix whose stored values in pattern order are `values`.
    ///
    /// Cached factors are reused unchanged when `values` matches the previous matrix bit for
    /// bit. Otherwise prior pivots are reused while each stays at least one-tenth of the
    /// largest entry in its column, falling back to threshold partial pivoting when violated.
    ///
    /// Condensing orderings eliminate condensed unknowns first and factor the reduced system,
    /// reusing factors when the reduced values match bit for bit.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] for structurally singular patterns (listing
    /// [undetermined](Ordering::undetermined) columns), zero pivots during elimination, or
    /// zero/non-finite condensation pivots. Returns [`LinearError::TooLarge`] when factor
    /// storage exceeds memory limits.
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

/// Maps singular column indices in `error` from the condensed pattern back to full-system
/// indices when `ordering` uses condensation.
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
    /// Returns [`LinearError::Singular`] when a pivot is too small for the matrix scale and
    /// yields a non-finite solution, naming the affected columns.
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
    /// Returns [`LinearError::Singular`] when a pivot is too small for the matrix scale and
    /// yields a non-finite solution, naming the affected columns.
    ///
    /// # Panics
    ///
    /// Panics if `right_hand_side` does not hold one value per row.
    pub fn solve_transpose(&mut self, right_hand_side: &mut [T]) -> Result<(), LinearError> {
        self.shared.solve_transpose(right_hand_side, self.workspace)
    }

    /// Returns a read-only [`Shared`] handle for solving across concurrent threads, each using
    /// a dedicated [`Workspace`], bit-identical to [`Factors::solve`].
    #[must_use]
    pub fn shared(&self) -> Shared<'a, T> {
        self.shared
    }
}

/// Shared read-only LU factors supporting concurrent solves from multiple threads with
/// per-thread [`Workspace`] buffers.
#[derive(Clone, Copy)]
pub struct Shared<'a, T: Scalar> {
    ordering: &'a Ordering,
    numeric: &'a Numeric<T>,
    values: &'a [T],
    /// Saved condensation values and multipliers used to reduce and expand solves.
    saved: &'a Saved<T>,
    /// Target thread count configured by [`SparseLu::spread`].
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
    /// Allocates a [`Workspace`] sized for this solver's factored pattern via
    /// [`Workspace::new`].
    #[must_use]
    pub fn workspace(&self) -> Workspace<T> {
        Workspace::new(self.ordering.factored().dimension())
    }

    /// Solves `A x = right_hand_side` and leaves `x` in `right_hand_side`, writing to
    /// `workspace` as it goes.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] when a pivot is too small for the matrix scale and
    /// yields a non-finite solution, naming the affected columns.
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
    /// Returns [`LinearError::Singular`] when a pivot is too small for the matrix scale and
    /// yields a non-finite solution, naming the affected columns.
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

/// Copies `from` into `into`, parallelizing across rayon in [`CHUNK`] slices when `threads >= 2`
/// and `from` spans multiple chunks into a matching-length buffer.
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

/// Runs [`finite`] in parallel [`CHUNK`] slices across rayon when `threads >= 2` and
/// `solution` spans multiple chunks.
fn finite_on<T: Scalar>(solution: &[T], threads: usize) -> Result<(), LinearError> {
    let all = |chunk: &[T]| chunk.iter().all(|&value| value.is_finite());
    if threads >= 2 && solution.len() > CHUNK && solution.par_chunks(CHUNK).all(all) {
        return Ok(());
    }
    finite(solution)
}

/// Checks that every entry in `solution` is finite, returning [`LinearError::Singular`] with
/// all non-finite column indices otherwise.
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
