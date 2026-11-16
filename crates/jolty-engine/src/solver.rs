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
            Ok(lu) => Ok(Factors {
                lu,
                names,
                refined: None,
            }),
            Err(error) => Err(named(error, names)),
        }
    }

    /// Factors the matrix whose stored values are `values`, as [`Solver::factor`] does,
    /// each solve with its factors then refined against it, as [`Factors::solve`] says.
    ///
    /// # Errors
    ///
    /// As [`Solver::factor`].
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
    lu: jolty_numerics::Factors<'a, T>,
    names: &'a [String],
    /// The matrix factored, its stored values and their pattern, and the buffers its
    /// refinement works in, when each solve is refined against it.
    refined: Option<(&'a [T], Arc<Pattern>, &'a mut Refinement<T>)>,
}

/// The buffers a refined solve works in, kept from one solve to the next so that none
/// allocates: a sweep solves thousands of small systems, each in a few hundred
/// nanoseconds.
#[derive(Clone, Debug, Default)]
pub(crate) struct Refinement<T> {
    given: Vec<T>,
    residual: Vec<T>,
    candidate: Vec<T>,
    left: Vec<T>,
    scale: Vec<f64>,
}

impl<'a, T: Scalar> Factors<'a, T> {
    /// Solves `A x = rhs` and leaves `x` in `rhs`.
    ///
    /// Factors kept for a matrix factored by [`Solver::factor_refined`] refine the
    /// solution against it by iterative refinement in working precision while that pays
    /// (Higham, *Accuracy and Stability of Numerical Algorithms*, 2002, §12.3): while the
    /// componentwise backward error `ω = maxᵢ |b − A x|ᵢ / (|A| |x| + |b|)ᵢ` of Oettli and
    /// Prager exceeds the machine epsilon, a correction solved from the residual is taken
    /// if it at least halves `ω`. Elimination with pivots chosen for another matrix, as a
    /// sweep keeps them from one frequency to the next, can lose the digits of a small
    /// unknown that the system determines well; one step of this refinement makes the
    /// solve componentwise backward stable (Skeel, *Mathematics of Computation* 35, 1980),
    /// which keeps them.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] when the solution is not finite, naming the
    /// unknowns whose values are not.
    pub(crate) fn solve(&mut self, rhs: &mut [T]) -> Result<(), SimulationError> {
        self.refine(rhs, false)
    }

    /// Solves the transposed system `Aᵀ x = rhs` and leaves `x` in `rhs`, refined as
    /// [`Factors::solve`] refines.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::Singular`] when the solution is not finite, naming the
    /// unknowns whose values are not.
    pub(crate) fn solve_transpose(&mut self, rhs: &mut [T]) -> Result<(), SimulationError> {
        self.refine(rhs, true)
    }

    /// Solves `A x = rhs`, or `Aᵀ x = rhs` if `transposed`, then refines as
    /// [`Factors::solve`] says.
    fn refine(&mut self, rhs: &mut [T], transposed: bool) -> Result<(), SimulationError> {
        let Some((matrix, pattern, work)) = self.refined.take() else {
            return self.plain(rhs, transposed);
        };
        let refined = self.refine_with(matrix, &pattern, work, rhs, transposed);
        self.refined = Some((matrix, pattern, work));
        refined
    }

    /// Refines as [`Factors::solve`] says, against the stored `matrix` of `pattern`, in
    /// the buffers of `work`.
    fn refine_with(
        &mut self,
        matrix: &[T],
        pattern: &Pattern,
        work: &mut Refinement<T>,
        rhs: &mut [T],
        transposed: bool,
    ) -> Result<(), SimulationError> {
        work.given.clear();
        work.given.extend_from_slice(rhs);
        self.plain(rhs, transposed)?;
        let mut error = backward_error(
            pattern,
            matrix,
            rhs,
            &work.given,
            transposed,
            &mut work.residual,
            &mut work.scale,
        );
        while error > f64::EPSILON {
            work.candidate.clear();
            work.candidate.extend_from_slice(&work.residual);
            self.plain(&mut work.candidate, transposed)?;
            for (value, correction) in work.candidate.iter_mut().zip(rhs.iter()) {
                *value += *correction;
            }
            let next = backward_error(
                pattern,
                matrix,
                &work.candidate,
                &work.given,
                transposed,
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

    /// Solves `A x = rhs`, or `Aᵀ x = rhs` if `transposed`, with the factors alone.
    fn plain(&mut self, rhs: &mut [T], transposed: bool) -> Result<(), SimulationError> {
        let solved = if transposed {
            self.lu.solve_transpose(rhs)
        } else {
            self.lu.solve(rhs)
        };
        solved.map_err(|error| named(error, self.names))
    }

    /// The factors, to solve with from several threads at once, each with a workspace of
    /// its own, as [`jolty_numerics::Factors::shared`] says.
    pub(crate) fn shared(&self) -> Shared<'a, T> {
        Shared {
            factors: self.lu.shared(),
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

/// The componentwise backward error of `x` as a solution of `A x = b`, or `Aᵀ x = b` if
/// `transposed`, `A` the stored `values` of `pattern`: `maxᵢ |rᵢ| / (|A| |x| + |b|)ᵢ`, the
/// residual `r = b − A x` left in `residual` and the denominators in `scale`. A row whose
/// denominator is zero has a zero residual too, and counts for nothing. Each `|·|` is a
/// number's [`Scalar::size`], a complex number's `|re| + |im|`, which bounds `|a x|` by
/// `|a| |x|` as the modulus does, within √2 of it, without a square root for each entry.
fn backward_error<T: Scalar>(
    pattern: &Pattern,
    values: &[T],
    x: &[T],
    b: &[T],
    transposed: bool,
    residual: &mut Vec<T>,
    scale: &mut Vec<f64>,
) -> f64 {
    scale.clear();
    scale.extend(b.iter().map(|value| value.size()));
    residual.clear();
    residual.extend_from_slice(b);
    for row in 0..x.len() {
        for &(slot, column) in pattern.row(row) {
            let (to, from) = if transposed {
                (column, row)
            } else {
                (row, column)
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
