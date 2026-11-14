//! The numerical kernels of jolty.
//!
//! Nothing here knows about circuits. A [`Pattern`] is the sparsity of a square system matrix,
//! collected once from the positions devices claim; its [`Ordering`] is analyzed once, and
//! [`SparseLu`] factors and solves matrices of that pattern, real or complex, sharing the
//! ordering and refactoring each new matrix with the pivots of the last;
//! [`integration`] holds the formulas that turn charges into currents over a time step; and
//! [`generalized_eigenvalues`] finds a dense matrix pencil's eigenvalues by the QZ algorithm,
//! and [`solve_in_range`] solves a small dense system, regular or not.

mod dense;
pub mod integration;
mod lu;
mod pattern;
mod pencil;

pub use dense::solve_in_range;
pub use lu::{Factors, LinearError, Ordering, Scalar, SparseLu, Statistics};
pub use num_complex::Complex64;
pub use pattern::{Entry, Pattern, PatternBuilder, Replay};
pub use pencil::{Eigenvalue, PencilError, generalized_eigenvalues};
