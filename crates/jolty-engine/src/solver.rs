//! The engine's sparse LU solvers: the simulator's, real, and those of the circuit
//! linearized at its operating point, complex, all sharing one ordering, whose failures name
//! the unknowns a singular matrix leaves undetermined.

use std::fmt;
use std::sync::Arc;

use jolty_numerics::{LinearError, Ordering, Pattern, Scalar, SparseLu, Workspace};

use crate::SimulationError;
use crate::newton::NAMED_UNKNOWNS;

/// A sparse LU solver of the circuit's matrices, whose failures name the unknowns.
#[derive(Clone)]
pub(crate) struct Solver<T: Scalar> {
    lu: SparseLu<T>,
    /// The name of every unknown, by index.
    names: Arc<[String]>,
}

impl<T: Scalar> Solver<T> {
    /// A solver of the matrices of `ordering`'s pattern, whose unknowns are `names`.
    pub(crate) fn new(ordering: Arc<Ordering>, names: Arc<[String]>) -> Self {
        Self {
            lu: SparseLu::new(ordering),
            names,
        }
    }

    /// Factors with `ordering`, of the same pattern, from now on, keeping the counts of
    /// work, as [`SparseLu::reorder`] does, unless it already does; and its dissected
    /// blocks on up to `threads` threads at once.
    pub(crate) fn order(&mut self, ordering: &Arc<Ordering>, threads: usize) {
        if !Arc::ptr_eq(self.lu.ordering(), ordering) {
            self.lu.reorder(Arc::clone(ordering));
        }
        self.lu.spread(threads);
    }

    /// A solver of the same matrices in another scalar, sharing this one's ordering, whose
    /// dissected blocks run on its calling thread alone: the analyses that share it spread
    /// their own solves.
    pub(crate) fn sharing<U: Scalar>(&self) -> Solver<U> {
        Solver::new(Arc::clone(self.lu.ordering()), Arc::clone(&self.names))
    }

    /// The ordering this solver factors with.
    pub(crate) fn ordering(&self) -> &Arc<Ordering> {
        self.lu.ordering()
    }

    /// The pattern of the matrices this solver factors.
    pub(crate) fn pattern(&self) -> &Arc<Pattern> {
        self.lu.pattern()
    }

    /// Takes on `other`'s ordering, pivots and factors, keeping its own counts of work and
    /// threads, as [`SparseLu::adopt`] does.
    pub(crate) fn adopt(&mut self, other: &Self) {
        self.lu.adopt(&other.lu);
    }

    /// Forgets every factorization, as [`SparseLu::reset`] does.
    pub(crate) fn reset(&mut self) {
        self.lu.reset();
    }

    /// How much work the solver has done.
    pub(crate) fn statistics(&self) -> jolty_numerics::Work {
        self.lu.statistics()
    }

    /// Whether this solver factors every matrix from now on as `other` would, bit for bit,
    /// as [`SparseLu::factors_alike`] says.
    pub(crate) fn factors_alike(&self, other: &Self) -> bool {
        self.lu.factors_alike(&other.lu)
    }

    /// Factors the matrix whose stored values are `values`, as [`SparseLu::factor`] does.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] when the matrix is singular, naming the
    /// unknowns it leaves undetermined, and [`SimulationError::TooLarge`] when its factors
    /// do not fit in memory.
    pub(crate) fn factor(&mut self, values: &[T]) -> Result<Factors<'_, T>, SimulationError> {
        let names = &self.names;
        match self.lu.factor(values) {
            Ok(factors) => Ok(Factors { factors, names }),
            Err(error) => Err(named(error, names)),
        }
    }

    /// The simulation error of the linear `error` of a system in the circuit's unknowns,
    /// naming the unknowns it leaves undetermined.
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

/// The LU factors of one of the circuit's matrices, ready to solve systems with it.
pub(crate) struct Factors<'a, T: Scalar> {
    factors: jolty_numerics::Factors<'a, T>,
    names: &'a [String],
}

impl<'a, T: Scalar> Factors<'a, T> {
    /// Solves `A x = rhs` and leaves `x` in `rhs`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] when the solution is not finite, naming the
    /// unknowns whose values are not.
    pub(crate) fn solve(&mut self, rhs: &mut [T]) -> Result<(), SimulationError> {
        self.factors
            .solve(rhs)
            .map_err(|error| named(error, self.names))
    }

    /// Solves the transposed system `Aᵀ x = rhs` and leaves `x` in `rhs`.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] when the solution is not finite, naming the
    /// unknowns whose values are not.
    pub(crate) fn solve_transpose(&mut self, rhs: &mut [T]) -> Result<(), SimulationError> {
        self.factors
            .solve_transpose(rhs)
            .map_err(|error| named(error, self.names))
    }

    /// The factors, to solve with from several threads at once, each with a workspace of
    /// its own, as [`jolty_numerics::Factors::shared`] says.
    pub(crate) fn shared(&self) -> Shared<'a, T> {
        Shared {
            factors: self.factors.shared(),
            names: self.names,
        }
    }
}

/// The LU factors of one of the circuit's matrices, which several threads solve with at
/// once, each with a workspace of its own.
#[derive(Clone, Copy)]
pub(crate) struct Shared<'a, T: Scalar> {
    factors: jolty_numerics::Shared<'a, T>,
    names: &'a [String],
}

impl<T: Scalar> Shared<'_, T> {
    /// A workspace for its solves, one for each thread that solves at once.
    pub(crate) fn workspace(&self) -> Workspace<T> {
        self.factors.workspace()
    }

    /// Solves `A x = rhs` and leaves `x` in `rhs`, as [`Factors::solve`] does, writing to
    /// `workspace` as it goes.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] when the solution is not finite, naming the
    /// unknowns whose values are not.
    pub(crate) fn solve(
        &self,
        rhs: &mut [T],
        workspace: &mut Workspace<T>,
    ) -> Result<(), SimulationError> {
        self.factors
            .solve(rhs, workspace)
            .map_err(|error| named(error, self.names))
    }

    /// Solves `Aᵀ x = rhs` and leaves `x` in `rhs`, as [`Factors::solve_transpose`] does,
    /// writing to `workspace` as it goes.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] when the solution is not finite, naming the
    /// unknowns whose values are not.
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

/// The simulation error of the linear `error`, naming the first [`NAMED_UNKNOWNS`] of the
/// columns it reports by the `names` of their unknowns.
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
