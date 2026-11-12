//! The numerical kernels of jolty.
//!
//! Nothing here knows about circuits. A [`Pattern`] is the sparsity of a square system matrix,
//! collected once from the positions devices claim; [`SparseLu`] factors and solves matrices
//! of that pattern, real or complex, reusing its symbolic analysis across every factorization;
//! [`integration`] holds the formulas that turn charges into currents over a time step; and
//! [`generalized_eigenvalues`] finds a dense matrix pencil's eigenvalues by the QZ algorithm.

pub mod integration;
mod lu;
mod pattern;
mod pencil;

pub use lu::{Factors, LinearError, Scalar, SparseLu};
pub use num_complex::Complex64;
pub use pattern::{Entry, Pattern, PatternBuilder};
pub use pencil::{Eigenvalue, PencilError, generalized_eigenvalues};
