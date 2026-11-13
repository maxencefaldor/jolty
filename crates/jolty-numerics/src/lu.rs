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

mod numeric;
mod ordering;

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};
use std::sync::Arc;

use faer::sparse::FaerError;
use num_complex::Complex64;

use self::numeric::Numeric;
pub use self::ordering::Ordering;
use crate::Pattern;

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
    /// Zero.
    const ZERO: Self;

    /// The value's magnitude: its absolute value, or a complex number's modulus.
    fn magnitude(self) -> f64;

    /// Whether the value is neither infinite nor NaN.
    fn is_finite(self) -> bool;

    /// Whether the value has the same bits as `other`: unlike `==`, a zero is not a
    /// negative zero, and a NaN can be identical to itself.
    fn is_identical(self, other: Self) -> bool;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;

    fn magnitude(self) -> f64 {
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
    const ZERO: Self = Self::new(0.0, 0.0);

    fn magnitude(self) -> f64 {
        self.norm()
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
pub struct Statistics {
    /// Factorizations that searched for pivots: the first, one after a failure, and one
    /// whose matrix made a pivot of the last too small, which searched from that pivot on.
    pub factorizations: usize,
    /// Factorizations that kept every pivot of the last, with no search.
    pub refactorizations: usize,
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
pub struct SparseLu<T: Scalar> {
    ordering: Arc<Ordering>,
    numeric: Numeric<T>,
    /// The values of the matrix last factored, which the solves read outside the diagonal
    /// blocks.
    factored: Vec<T>,
    /// Whether the factors are those of `factored`.
    current: bool,
    /// The solves' workspace, by row of the ordered matrix and by position.
    workspace: (Vec<T>, Vec<T>),
    statistics: Statistics,
}

impl<T: Scalar> SparseLu<T> {
    /// A solver for the matrices of `ordering`'s pattern.
    #[must_use]
    pub fn new(ordering: Arc<Ordering>) -> Self {
        let dimension = ordering.pattern().dimension();
        Self {
            numeric: Numeric::new(dimension),
            ordering,
            factored: Vec::new(),
            current: false,
            workspace: (vec![T::ZERO; dimension], vec![T::ZERO; dimension]),
            statistics: Statistics::default(),
        }
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
    pub fn statistics(&self) -> Statistics {
        self.statistics
    }

    /// The entries the solves read: those of `L` below its unit diagonal, those of `U` on
    /// and above its diagonal, and the matrix's own entries above its diagonal blocks, as
    /// the last factorization left them. The fill is their number less the matrix's.
    #[must_use]
    pub fn entries(&self) -> usize {
        self.numeric.entries() + self.ordering.entries_above()
    }

    /// Factors the matrix whose stored values, in the pattern's order, are `values`.
    ///
    /// The factors of the last matrix are kept when every value is bitwise identical to its
    /// own. Otherwise the pivots of the last factorization are kept while each stays at
    /// least a tenth of the largest entry below it in its column, and the pivots are
    /// searched for from the first that does not, each the diagonal entry of the ordering
    /// where it is that large, and else the largest.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] for every matrix of a structurally singular
    /// pattern, naming its [undetermined](Ordering::undetermined) columns, and for a matrix
    /// with a column that has no nonzero pivot left once the columns before it are
    /// eliminated, naming that column; and [`LinearError::TooLarge`] when the factors do not
    /// fit in memory.
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
            self.factored.clear();
            self.factored.extend_from_slice(values);
            if self.numeric.factor(&self.ordering, values)? {
                self.statistics.factorizations += 1;
            } else {
                self.statistics.refactorizations += 1;
            }
            self.current = true;
        }
        Ok(Factors {
            ordering: &self.ordering,
            numeric: &self.numeric,
            values: &self.factored,
            workspace: &mut self.workspace,
        })
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
    ordering: &'a Ordering,
    numeric: &'a Numeric<T>,
    values: &'a [T],
    workspace: &'a mut (Vec<T>, Vec<T>),
}

impl<T: Scalar> fmt::Debug for Factors<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Factors")
            .field("rows", &self.ordering.pattern().dimension())
            .finish_non_exhaustive()
    }
}

impl<T: Scalar> Factors<'_, T> {
    /// Solves `A x = rhs` and leaves `x` in `rhs`.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] when the solution is not finite, which a pivot
    /// too small for the matrix's range causes, naming the columns whose values are not.
    ///
    /// # Panics
    ///
    /// Panics if `rhs` does not hold one value per row.
    pub fn solve(&mut self, rhs: &mut [T]) -> Result<(), LinearError> {
        assert_eq!(
            rhs.len(),
            self.ordering.pattern().dimension(),
            "one right-hand side value per row"
        );
        let (by_row, by_position) = &mut *self.workspace;
        self.numeric
            .solve(self.ordering, self.values, rhs, (by_row, by_position));
        finite(rhs)
    }

    /// Solves `Aᵀ x = rhs`, the transposed system, and leaves `x` in `rhs`: the adjoint
    /// system whose solution weighs every equation's effect on one unknown.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] when the solution is not finite, which a pivot
    /// too small for the matrix's range causes, naming the columns whose values are not.
    ///
    /// # Panics
    ///
    /// Panics if `rhs` does not hold one value per row.
    pub fn solve_transpose(&mut self, rhs: &mut [T]) -> Result<(), LinearError> {
        assert_eq!(
            rhs.len(),
            self.ordering.pattern().dimension(),
            "one right-hand side value per row"
        );
        let (by_row, by_position) = &mut *self.workspace;
        self.numeric
            .solve_transpose(self.ordering, self.values, rhs, (by_row, by_position));
        finite(rhs)
    }
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
