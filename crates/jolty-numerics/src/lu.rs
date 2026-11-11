use std::fmt;
use std::sync::Arc;

use dyn_stack::{MemBuffer, MemStack};
use faer::sparse::linalg::LuError;
use faer::sparse::linalg::lu::{
    LuRef, LuSymbolicParams, NumericLu, SymbolicLu, factorize_symbolic_lu,
};
use faer::sparse::{FaerError, SparseColMatRef};
use faer::traits::ComplexField;
use faer::{Conj, MatMut, Par, Spec};
use num_complex::Complex64;

use crate::Pattern;

/// A scalar the sparse solver works in: `f64` for DC and transient analyses, [`Complex64`]
/// for small-signal analyses.
pub trait Scalar: ComplexField + Copy {
    /// Whether the value is neither infinite nor NaN.
    fn is_finite(self) -> bool;
}

impl Scalar for f64 {
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
}

impl Scalar for Complex64 {
    fn is_finite(self) -> bool {
        Complex64::is_finite(self)
    }
}

/// Why a linear system could not be solved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LinearError {
    /// The matrix is singular, structurally or numerically.
    #[error("the system matrix is singular")]
    Singular,
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

impl From<LuError> for LinearError {
    fn from(error: LuError) -> Self {
        match error {
            LuError::SymbolicSingular { .. } => Self::Singular,
            LuError::Generic(_) => Self::TooLarge,
        }
    }
}

/// A sparse LU solver for matrices of one [`Pattern`].
///
/// Construction runs the symbolic analysis, a fill-reducing column ordering, once. Every
/// [`SparseLu::factor`] then reuses it and its workspaces, so the Newton iterations of a whole
/// simulation allocate nothing.
pub struct SparseLu<T: Scalar> {
    pattern: Arc<Pattern>,
    symbolic: SymbolicLu<usize>,
    numeric: NumericLu<usize, T>,
    factor_workspace: MemBuffer,
    solve_workspace: MemBuffer,
}

impl<T: Scalar> SparseLu<T> {
    /// Analyzes `pattern` and allocates the workspaces its factorizations need.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::TooLarge`] when the analysis or its workspaces do not fit.
    pub fn new(pattern: Arc<Pattern>) -> Result<Self, LinearError> {
        let symbolic =
            factorize_symbolic_lu(pattern.structure().as_ref(), LuSymbolicParams::default())?;
        let factor_workspace = MemBuffer::try_new(
            symbolic.factorize_numeric_lu_scratch::<T>(Par::Seq, Spec::default()),
        )?;
        let solve_workspace =
            MemBuffer::try_new(symbolic.solve_in_place_scratch::<T>(1, Par::Seq))?;
        Ok(Self {
            pattern,
            symbolic,
            numeric: NumericLu::new(),
            factor_workspace,
            solve_workspace,
        })
    }

    /// The pattern of the matrices this solver factors.
    #[must_use]
    pub fn pattern(&self) -> &Arc<Pattern> {
        &self.pattern
    }

    /// Factors the matrix whose stored values, in the pattern's order, are `values`.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] when no pivot can be found for some column.
    ///
    /// # Panics
    ///
    /// Panics if `values` does not hold one value per stored position of the pattern.
    pub fn factor(&mut self, values: &[T]) -> Result<Factors<'_, T>, LinearError> {
        assert_eq!(
            values.len(),
            self.pattern.len(),
            "one value per stored position"
        );
        let matrix = SparseColMatRef::new(self.pattern.structure().as_ref(), values);
        self.symbolic.factorize_numeric_lu(
            &mut self.numeric,
            matrix,
            Par::Seq,
            MemStack::new(&mut self.factor_workspace),
            Spec::default(),
        )?;
        Ok(Factors {
            factors: LuRef::new_unchecked(&self.symbolic, &self.numeric),
            workspace: &mut self.solve_workspace,
        })
    }
}

impl<T: Scalar> fmt::Debug for SparseLu<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseLu")
            .field("pattern", &self.pattern)
            .finish_non_exhaustive()
    }
}

/// The LU factors of one matrix, ready to solve systems with it.
pub struct Factors<'a, T: Scalar> {
    factors: LuRef<'a, usize, T>,
    workspace: &'a mut MemBuffer,
}

impl<T: Scalar> fmt::Debug for Factors<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Factors")
            .field("rows", &self.factors.symbolic().nrows())
            .finish_non_exhaustive()
    }
}

impl<T: Scalar> Factors<'_, T> {
    /// Solves `A x = rhs` and leaves `x` in `rhs`.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] when the solution is not finite, which a zero pivot
    /// causes.
    ///
    /// # Panics
    ///
    /// Panics if `rhs` does not hold one value per row.
    pub fn solve(&mut self, rhs: &mut [T]) -> Result<(), LinearError> {
        let rows = self.factors.symbolic().nrows();
        assert_eq!(rhs.len(), rows, "one right-hand side value per row");
        self.factors.solve_in_place_with_conj(
            Conj::No,
            MatMut::from_column_major_slice_mut(rhs, rows, 1),
            Par::Seq,
            MemStack::new(self.workspace),
        );
        finite(rhs)
    }

    /// Solves `Aᵀ x = rhs`, the transposed system, and leaves `x` in `rhs`: the adjoint
    /// system whose solution weighs every equation's effect on one unknown.
    ///
    /// # Errors
    ///
    /// Returns [`LinearError::Singular`] when the solution is not finite, which a zero pivot
    /// causes.
    ///
    /// # Panics
    ///
    /// Panics if `rhs` does not hold one value per row.
    pub fn solve_transpose(&mut self, rhs: &mut [T]) -> Result<(), LinearError> {
        let rows = self.factors.symbolic().nrows();
        assert_eq!(rhs.len(), rows, "one right-hand side value per row");
        self.factors.solve_transpose_in_place_with_conj(
            Conj::No,
            MatMut::from_column_major_slice_mut(rhs, rows, 1),
            Par::Seq,
            MemStack::new(self.workspace),
        );
        finite(rhs)
    }
}

/// `Ok` when every value of a solution is finite.
fn finite<T: Scalar>(solution: &[T]) -> Result<(), LinearError> {
    if solution.iter().all(|&value| value.is_finite()) {
        Ok(())
    } else {
        Err(LinearError::Singular)
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::PatternBuilder;

    /// A dense matrix stored with a full pattern, and its pattern.
    fn dense(matrix: &[Vec<f64>]) -> (Arc<Pattern>, Vec<f64>) {
        let n = matrix.len();
        let mut builder = PatternBuilder::new();
        let entries: Vec<_> = (0..n)
            .flat_map(|row| (0..n).map(move |column| (row, column)))
            .map(|(row, column)| (builder.entry(row, column), matrix[row][column]))
            .collect();
        let pattern = Arc::new(builder.finish(n));
        let mut values = vec![0.0; pattern.len()];
        for (entry, value) in entries {
            values[pattern.slot(entry)] += value;
        }
        (pattern, values)
    }

    #[test]
    fn solves_a_system_that_needs_pivoting() {
        let (pattern, values) = dense(&[vec![0.0, 1.0], vec![2.0, 3.0]]);
        let mut lu = SparseLu::new(pattern).unwrap();
        let mut rhs = [1.0, 8.0];
        lu.factor(&values).unwrap().solve(&mut rhs).unwrap();
        assert!(
            (rhs[0] - 2.5).abs() < 1e-15 && (rhs[1] - 1.0).abs() < 1e-15,
            "{rhs:?}"
        );
    }

    #[test]
    fn solves_the_transposed_system() {
        let (pattern, values) = dense(&[vec![0.0, 1.0], vec![2.0, 3.0]]);
        let mut lu = SparseLu::new(pattern).unwrap();
        // [[0, 2], [1, 3]] x = [4, 7] has x = [1, 2].
        let mut rhs = [4.0, 7.0];
        lu.factor(&values)
            .unwrap()
            .solve_transpose(&mut rhs)
            .unwrap();
        assert!(
            (rhs[0] - 1.0).abs() < 1e-15 && (rhs[1] - 2.0).abs() < 1e-15,
            "{rhs:?}"
        );
    }

    #[test]
    fn solves_complex_systems() {
        let mut builder = PatternBuilder::new();
        let entry = builder.entry(0, 0);
        let pattern = Arc::new(builder.finish(1));
        let mut values = vec![Complex64::default(); 1];
        values[pattern.slot(entry)] = Complex64::new(0.0, 2.0);
        let mut rhs = [Complex64::new(4.0, 0.0)];
        SparseLu::new(pattern)
            .unwrap()
            .factor(&values)
            .unwrap()
            .solve(&mut rhs)
            .unwrap();
        assert_eq!(rhs[0], Complex64::new(0.0, -2.0));
    }

    #[test]
    fn reports_singular_matrices() {
        let (pattern, values) = dense(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        let mut lu = SparseLu::new(pattern).unwrap();
        let result = lu
            .factor(&values)
            .and_then(|mut factors| factors.solve(&mut [1.0, 1.0]));
        assert_eq!(result, Err(LinearError::Singular));
    }

    proptest! {
        /// A diagonally dominant matrix is nonsingular, and its solution reproduces the
        /// right-hand side.
        #[test]
        fn solutions_satisfy_the_system(
            seed in proptest::collection::vec(-1.0..1.0f64, 36),
            rhs in proptest::collection::vec(-10.0..10.0f64, 6),
        ) {
            let n = 6;
            // Off-diagonal entries of a row sum to at most 5 in magnitude, so a diagonal above 11
            // makes the matrix strictly diagonally dominant.
            let matrix: Vec<Vec<f64>> = (0..n)
                .map(|row| (0..n).map(|column| {
                    let value = seed[row * n + column];
                    if row == column { value + 12.0 } else { value }
                }).collect())
                .collect();
            let (pattern, values) = dense(&matrix);
            let mut solution = rhs.clone();
            SparseLu::new(pattern).unwrap().factor(&values).unwrap().solve(&mut solution).unwrap();
            for row in 0..n {
                let product: f64 =
                    matrix[row].iter().zip(&solution).map(|(entry, value)| entry * value).sum();
                prop_assert!((product - rhs[row]).abs() < 1e-9);
            }
        }
    }
}
