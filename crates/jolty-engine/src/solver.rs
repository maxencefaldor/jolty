//! The engine's sparse LU solvers: the simulator's, real, and those of the circuit
//! linearized at its operating point, complex, all sharing one ordering, whose failures name
//! the unknowns a singular matrix leaves undetermined.

use std::fmt;
use std::sync::Arc;

use jolty_numerics::{LinearError, Ordering, Pattern, Scalar, SparseLu};

use crate::SimulationError;
use crate::newton::NAMED_UNKNOWNS;

/// A sparse LU solver of the circuit's matrices, whose failures name the unknowns.
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

    /// A solver of the same matrices in another scalar, sharing this one's ordering.
    pub(crate) fn sharing<U: Scalar>(&self) -> Solver<U> {
        Solver::new(Arc::clone(self.lu.ordering()), Arc::clone(&self.names))
    }

    /// The pattern of the matrices this solver factors.
    pub(crate) fn pattern(&self) -> &Arc<Pattern> {
        self.lu.pattern()
    }

    /// How much work the solver has done.
    #[cfg(test)]
    pub(crate) fn statistics(&self) -> jolty_numerics::Statistics {
        self.lu.statistics()
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

impl<T: Scalar> Factors<'_, T> {
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
