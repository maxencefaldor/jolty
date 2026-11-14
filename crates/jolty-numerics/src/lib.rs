//! The numerical kernels of jolty.
//!
//! Nothing here knows about circuits. A [`Pattern`] is the sparsity of a square system matrix,
//! collected once from the positions devices claim; its [`Ordering`] is analyzed once, and
//! [`SparseLu`] factors and solves matrices of that pattern, real or complex, sharing the
//! ordering and refactoring each new matrix with the pivots of the last, the blocks an
//! ordering dissects on any number of threads, the same bits on each; a [`GatherMap`]
//! assembles a system's values from the contributions a slab holds, each in a fixed order
//! however many threads gather them; [`reduce`] sums many terms in one fixed order on any
//! number of threads; [`integration`] holds the formulas that turn charges
//! into currents over a time step; and [`generalized_eigenvalues`] finds a dense matrix pencil's eigenvalues by the QZ algorithm,
//! and [`solve_in_range`] solves a small dense system, regular or not.

mod dense;
mod gather;
pub mod integration;
mod lu;
mod pattern;
mod pencil;
pub mod reduce;

pub use dense::solve_in_range;
pub use gather::{Fold, GatherMap, UNMAPPED};
pub use lu::{Factors, LinearError, Ordering, Scalar, Shared, SparseLu, Statistics, Workspace};
pub use num_complex::Complex64;
pub use pattern::{Entry, Pattern, PatternBuilder};
pub use pencil::{Eigenvalue, PencilError, generalized_eigenvalues};
