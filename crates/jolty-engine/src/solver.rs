//! Sparse LU solvers for real DC/transient systems and complex linearized small-signal
//! systems, sharing symbolic ordering and reporting undetermined unknown names on failure.

use std::fmt;
use std::sync::Arc;

use jolty_numerics::{LinearError, Ordering, Pattern, Scalar, SparseLu, Workspace};

use crate::SimulationError;
use crate::newton::NAMED_UNKNOWNS;

/// Sparse LU solver for circuit matrices that annotates singularity errors with unknown
/// names.
#[derive(Clone)]
pub(crate) struct Solver<T: Scalar> {
    lu: SparseLu<T>,
    /// Unknown names indexed by matrix column.
    names: Arc<[String]>,
}

impl<T: Scalar> Solver<T> {
    /// Creates a [`Solver`] for `ordering` with unknown `names`.
    pub(crate) fn new(ordering: Arc<Ordering>, names: Arc<[String]>) -> Self {
        Self {
            lu: SparseLu::new(ordering),
            names,
        }
    }

    /// Switches to `ordering` when distinct via [`SparseLu::reorder`] (preserving work
    /// counts) and sets the factorization thread limit to `threads`.
    pub(crate) fn order(&mut self, ordering: &Arc<Ordering>, threads: usize) {
        if !Arc::ptr_eq(self.lu.ordering(), ordering) {
            self.lu.reorder(Arc::clone(ordering));
        }
        self.lu.spread(threads);
    }

    /// Creates a single-threaded [`Solver`] for scalar type `U` sharing this solver's
    /// symbolic ordering (used by analyses that parallelize across solves instead).
    pub(crate) fn sharing<U: Scalar>(&self) -> Solver<U> {
        Solver::new(Arc::clone(self.lu.ordering()), Arc::clone(&self.names))
    }

    /// Returns the active symbolic [`Ordering`].
    pub(crate) fn ordering(&self) -> &Arc<Ordering> {
        self.lu.ordering()
    }

    /// Returns the sparse matrix [`Pattern`] factored by this solver.
    pub(crate) fn pattern(&self) -> &Arc<Pattern> {
        self.lu.pattern()
    }

    /// Adopts `other`'s ordering, pivots, and numeric factors via [`SparseLu::adopt`]
    /// while preserving local work counters and thread count.
    pub(crate) fn adopt(&mut self, other: &Self) {
        self.lu.adopt(&other.lu);
    }

    /// Clears cached factorizations via [`SparseLu::reset`].
    pub(crate) fn reset(&mut self) {
        self.lu.reset();
    }

    /// Returns cumulative factorization work counters.
    pub(crate) fn statistics(&self) -> jolty_numerics::Work {
        self.lu.statistics()
    }

    /// Returns whether this solver and `other` hold identical pivot state per
    /// [`SparseLu::factors_alike`].
    pub(crate) fn factors_alike(&self, other: &Self) -> bool {
        self.lu.factors_alike(&other.lu)
    }

    /// Factors sparse matrix `values` via [`SparseLu::factor`].
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] naming undetermined unknowns, or
    /// [`SimulationError::TooLarge`] when factor storage overflows memory limits.
    pub(crate) fn factor(&mut self, values: &[T]) -> Result<Factors<'_, T>, SimulationError> {
        let names = &self.names;
        match self.lu.factor(values) {
            Ok(lu) => Ok(Factors {
                lu,
                names,
                refined: None,
            }),
            Err(error) => Err(named(error, names)),
        }
    }

    /// Factors sparse matrix `values` like [`Solver::factor`] and attaches `refinement` so
    /// subsequent solves run iterative refinement per [`Factors::solve`].
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Solver::factor`].
    pub(crate) fn factor_refined<'a>(
        &'a mut self,
        values: &'a [T],
        refinement: &'a mut Refinement<T>,
    ) -> Result<Factors<'a, T>, SimulationError> {
        let pattern = Arc::clone(self.pattern());
        let mut factors = self.factor(values)?;
        factors.refined = Some((values, pattern, refinement));
        Ok(factors)
    }

    /// Converts linear solver `error` into a [`SimulationError`] naming undetermined
    /// unknowns.
    pub(crate) fn name(&self, error: LinearError) -> SimulationError {
        named(error, &self.names)
    }
}

impl<T: Scalar> fmt::Debug for Solver<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Solver")
            .field("lu", &self.lu)
            .finish_non_exhaustive()
    }
}

/// Factored sparse LU handle ready to solve linear systems in circuit unknowns.
pub(crate) struct Factors<'a, T: Scalar> {
    lu: jolty_numerics::Factors<'a, T>,
    names: &'a [String],
    /// Original matrix values, pattern, and scratch buffers used when iterative refinement
    /// is enabled.
    refined: Option<(&'a [T], Arc<Pattern>, &'a mut Refinement<T>)>,
}

/// Reusable scratch buffers for working-precision iterative refinement, avoiding per-solve
/// allocations during frequency sweeps.
#[derive(Clone, Debug, Default)]
pub(crate) struct Refinement<T> {
    given: Vec<T>,
    residual: Vec<T>,
    candidate: Vec<T>,
    left: Vec<T>,
    scale: Vec<f64>,
}

/// Linear system orientation selecting `A` or `Aᵀ`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Direct,
    Transposed,
}

impl<'a, T: Scalar> Factors<'a, T> {
    /// Solves `A x = rhs` in place, writing `x` into `rhs`.
    ///
    /// When created via [`Solver::factor_refined`], runs working-precision iterative
    /// refinement (Higham, *Accuracy and Stability of Numerical Algorithms*, 2002, §12.3).
    /// Refinement iterates while the Oettli-Prager componentwise backward error
    /// `ω = maxᵢ |b - A x|ᵢ / (|A| |x| + |b|)ᵢ` exceeds machine epsilon and each step at
    /// least halves `ω`. One step restores componentwise backward stability when reused
    /// pivots lose accuracy on small unknowns (Skeel, *Mathematics of Computation* 35,
    /// 1980).
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] naming non-finite solution unknowns.
    pub(crate) fn solve(&mut self, rhs: &mut [T]) -> Result<(), SimulationError> {
        self.refine(rhs, Side::Direct)
    }

    /// Solves `Aᵀ x = rhs` in place, applying iterative refinement matching
    /// [`Factors::solve`].
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] naming non-finite solution unknowns.
    pub(crate) fn solve_transpose(&mut self, rhs: &mut [T]) -> Result<(), SimulationError> {
        self.refine(rhs, Side::Transposed)
    }

    /// Solves `A x = rhs` or `Aᵀ x = rhs` per `side` and runs iterative refinement when
    /// enabled.
    fn refine(&mut self, rhs: &mut [T], side: Side) -> Result<(), SimulationError> {
        let Some((matrix, pattern, work)) = self.refined.take() else {
            return self.plain(rhs, side);
        };
        let refined = self.refine_with(matrix, &pattern, work, rhs, side);
        self.refined = Some((matrix, pattern, work));
        refined
    }

    /// Performs working-precision iterative refinement against `matrix` and `pattern`
    /// using scratch buffers in `work`.
    fn refine_with(
        &mut self,
        matrix: &[T],
        pattern: &Pattern,
        work: &mut Refinement<T>,
        rhs: &mut [T],
        side: Side,
    ) -> Result<(), SimulationError> {
        work.given.clear();
        work.given.extend_from_slice(rhs);
        self.plain(rhs, side)?;
        let mut error = backward_error(
            pattern,
            matrix,
            rhs,
            &work.given,
            side,
            &mut work.residual,
            &mut work.scale,
        );
        while error > f64::EPSILON {
            work.candidate.clear();
            work.candidate.extend_from_slice(&work.residual);
            self.plain(&mut work.candidate, side)?;
            for (value, correction) in work.candidate.iter_mut().zip(rhs.iter()) {
                *value += *correction;
            }
            let next = backward_error(
                pattern,
                matrix,
                &work.candidate,
                &work.given,
                side,
                &mut work.left,
                &mut work.scale,
            );
            if next > error / 2.0 {
                break;
            }
            rhs.copy_from_slice(&work.candidate);
            std::mem::swap(&mut work.residual, &mut work.left);
            error = next;
        }
        Ok(())
    }

    /// Solves `A x = rhs` or `Aᵀ x = rhs` per `side` without iterative refinement.
    fn plain(&mut self, rhs: &mut [T], side: Side) -> Result<(), SimulationError> {
        let solved = match side {
            Side::Direct => self.lu.solve(rhs),
            Side::Transposed => self.lu.solve_transpose(rhs),
        };
        solved.map_err(|error| named(error, self.names))
    }

    /// Borrows read-only [`Shared`] factors for concurrent multi-threaded solves per
    /// [`jolty_numerics::Factors::shared`].
    pub(crate) fn shared(&self) -> Shared<'a, T> {
        Shared {
            factors: self.lu.shared(),
            names: self.names,
        }
    }
}

/// Shared read-only sparse LU factors supporting concurrent solves with per-thread
/// workspaces.
#[derive(Clone, Copy)]
pub(crate) struct Shared<'a, T: Scalar> {
    factors: jolty_numerics::Shared<'a, T>,
    names: &'a [String],
}

impl<T: Scalar> Shared<'_, T> {
    /// Allocates a thread-local solve [`Workspace`] sized for these factors.
    pub(crate) fn workspace(&self) -> Workspace<T> {
        self.factors.workspace()
    }

    /// Solves `A x = rhs` in place using `workspace`, matching [`Factors::solve`].
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] naming non-finite solution unknowns.
    pub(crate) fn solve(
        &self,
        rhs: &mut [T],
        workspace: &mut Workspace<T>,
    ) -> Result<(), SimulationError> {
        self.factors
            .solve(rhs, workspace)
            .map_err(|error| named(error, self.names))
    }

    /// Solves `Aᵀ x = rhs` in place using `workspace`, matching [`Factors::solve_transpose`].
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] naming non-finite solution unknowns.
    pub(crate) fn solve_transpose(
        &self,
        rhs: &mut [T],
        workspace: &mut Workspace<T>,
    ) -> Result<(), SimulationError> {
        self.factors
            .solve_transpose(rhs, workspace)
            .map_err(|error| named(error, self.names))
    }
}

/// Computes the componentwise backward error `maxᵢ |rᵢ| / (|A| |x| + |b|)ᵢ` for `x`
/// solving `A x = b` (or `Aᵀ x = b`), storing `r = b - A x` in `residual` and denominators
/// in `scale`.
///
/// Rows with zero denominator also have zero residual and contribute zero. Magnitudes use
/// [`Scalar::size`] (`|re| + |im|` for complex numbers), bounding `|a x|` within `√2`
/// without per-entry square roots.
fn backward_error<T: Scalar>(
    pattern: &Pattern,
    values: &[T],
    x: &[T],
    b: &[T],
    side: Side,
    residual: &mut Vec<T>,
    scale: &mut Vec<f64>,
) -> f64 {
    scale.clear();
    scale.extend(b.iter().map(|value| value.size()));
    residual.clear();
    residual.extend_from_slice(b);
    for row in 0..x.len() {
        for &(slot, column) in pattern.row(row) {
            let (to, from) = match side {
                Side::Direct => (row, column),
                Side::Transposed => (column, row),
            };
            let term = values[slot] * x[from];
            residual[to] -= term;
            scale[to] += values[slot].size() * x[from].size();
        }
    }
    residual
        .iter()
        .zip(scale.iter())
        .filter(|&(_, &scale)| scale > 0.0)
        .fold(0.0, |most: f64, (value, scale)| {
            most.max(value.size() / scale)
        })
}

/// Converts linear solver `error` into a [`SimulationError`], naming up to
/// [`NAMED_UNKNOWNS`] undetermined columns from `names`.
fn named(error: LinearError, names: &[String]) -> SimulationError {
    match error {
        LinearError::Singular { columns } => SimulationError::Singular {
            unknowns: columns
                .into_iter()
                .take(NAMED_UNKNOWNS)
                .map(|column| names[column].clone())
                .collect(),
        },
        LinearError::TooLarge => SimulationError::TooLarge,
    }
}
