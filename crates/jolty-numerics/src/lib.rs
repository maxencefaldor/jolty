//! Circuit-agnostic numerical kernels for `jolty`.
//!
//! - [`Pattern`] captures the sparsity structure of a square system matrix from claimed
//!   entry positions, and [`Ordering`] analyzes that sparsity once for reuse.
//! - [`SparseLu`] factors and solves real or complex matrices sharing an [`Ordering`],
//!   reusing pivots across refactorizations and dissecting independent blocks deterministically
//!   across any thread count.
//! - [`GatherMap`] assembles system vectors and matrix entries from slab contributions in a
//!   fixed, thread-count-independent order.
//! - [`reduce`] performs deterministic chunked reductions across any thread count.
//! - [`integration`] provides multistep time-integration formulas turning charges into
//!   currents over variable time steps.
//! - [`generalized_eigenvalues`] computes eigenvalues of dense matrix pencils via the QZ
//!   algorithm, while [`solve_in_range`] and [`solve_each_in_range`] solve small regular or
//!   rank-deficient dense linear systems.

mod dense;
mod gather;
pub mod integration;
mod lu;
mod pattern;
mod pencil;
pub mod reduce;

pub use dense::{solve_each_in_range, solve_in_range};
pub use gather::{Fold, GatherMap, UNMAPPED};
pub use lu::{
    Condensation, Dissecting, Factors, LinearError, Ordering, Scalar, Shared, SparseLu, Work,
    Workspace,
};
pub use num_complex::Complex64;
pub use pattern::{Entry, Pattern, PatternBuilder};
pub use pencil::{Eigenvalue, PencilError, generalized_eigenvalues};
