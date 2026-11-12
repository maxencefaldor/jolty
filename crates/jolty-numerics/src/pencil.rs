//! The generalized eigenvalues of dense matrix pencils, by the QZ algorithm.

use dyn_stack::{MemBuffer, MemStack};
use faer::diag::Diag;
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{gevd_cplx, gevd_scratch};
use faer::{Mat, Par, Spec};
use num_complex::Complex64;

/// A generalized eigenvalue of a pencil `(A, B)`, `alpha / beta`, kept as its pair so that
/// an infinite one, whose `beta` is zero, keeps its place.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eigenvalue {
    /// The numerator.
    pub alpha: Complex64,
    /// The denominator.
    pub beta: Complex64,
}

/// Why a pencil's eigenvalues could not be found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PencilError {
    /// The QZ iteration reached its limit before every eigenvalue converged.
    #[error("the QZ iteration did not converge")]
    Unconverged,
    /// The pencil exceeds the memory available.
    #[error("the pencil is too large")]
    TooLarge,
}

/// The generalized eigenvalues of the pencil `(a, b)`, the `λ` for which `a v = λ b v` with
/// `v` nonzero, of two dense square matrices of `dimension` rows, in column-major order.
///
/// # Errors
///
/// Returns [`PencilError::Unconverged`] when the QZ iteration does not converge, and
/// [`PencilError::TooLarge`] when its workspace cannot be allocated.
///
/// # Panics
///
/// Panics unless each matrix holds `dimension²` values.
pub fn generalized_eigenvalues(
    dimension: usize,
    a: &[f64],
    b: &[f64],
) -> Result<Vec<Eigenvalue>, PencilError> {
    let n = dimension;
    assert_eq!(a.len(), n * n, "a square matrix of the dimension");
    assert_eq!(b.len(), n * n, "a square matrix of the dimension");
    if n == 0 {
        return Ok(Vec::new());
    }
    // The complex QZ iteration, which gives each eigenvalue of a conjugate pair its own
    // numerator and denominator.
    let entry =
        |matrix: &[f64], row: usize, column: usize| Complex64::new(matrix[column * n + row], 0.0);
    let mut a = Mat::from_fn(n, n, |row, column| entry(a, row, column));
    let mut b = Mat::from_fn(n, n, |row, column| entry(b, row, column));
    let (mut alpha, mut beta) = (Diag::zeros(n), Diag::zeros(n));
    let scratch = gevd_scratch::<Complex64>(
        n,
        ComputeEigenvectors::No,
        ComputeEigenvectors::No,
        Par::Seq,
        Spec::default(),
    );
    let mut buffer = MemBuffer::try_new(scratch).map_err(|_| PencilError::TooLarge)?;
    gevd_cplx(
        a.as_mut(),
        b.as_mut(),
        alpha.as_mut(),
        beta.as_mut(),
        None,
        None,
        Par::Seq,
        MemStack::new(&mut buffer),
        Spec::default(),
    )
    .map_err(|_| PencilError::Unconverged)?;
    Ok((0..n)
        .map(|index| Eigenvalue {
            alpha: alpha[index],
            beta: beta[index],
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The finite eigenvalues, sorted by their imaginary parts, then their real ones.
    fn finite(eigenvalues: &[Eigenvalue]) -> Vec<Complex64> {
        let mut found: Vec<Complex64> = eigenvalues
            .iter()
            .filter(|eigenvalue| eigenvalue.beta.norm() > 1e-12 * eigenvalue.alpha.norm())
            .map(|eigenvalue| eigenvalue.alpha / eigenvalue.beta)
            .collect();
        found.sort_by(|x, y| x.im.total_cmp(&y.im).then(x.re.total_cmp(&y.re)));
        found
    }

    #[test]
    fn finds_real_complex_and_infinite_eigenvalues() {
        // A = [[0, -2, 0], [1, 0, 0], [0, 0, 3]], B = diag(1, 1, 0): λ = ±j√2, and one
        // infinite, as B is singular.
        let a = [0.0, 1.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 3.0];
        let b = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let eigenvalues = generalized_eigenvalues(3, &a, &b).expect("converges");
        assert_eq!(eigenvalues.len(), 3);
        let found = finite(&eigenvalues);
        assert_eq!(found.len(), 2, "{eigenvalues:?}");
        let root = 2.0_f64.sqrt();
        assert!(
            (found[0] - Complex64::new(0.0, -root)).norm() < 1e-12,
            "{found:?}"
        );
        assert!(
            (found[1] - Complex64::new(0.0, root)).norm() < 1e-12,
            "{found:?}"
        );
        // A diagonal pencil: λ = 2 and -3.
        let a = [2.0, 0.0, 0.0, -6.0];
        let b = [1.0, 0.0, 0.0, 2.0];
        let found = finite(&generalized_eigenvalues(2, &a, &b).expect("converges"));
        assert!((found[0] + 3.0).norm() < 1e-12 && (found[1] - 2.0).norm() < 1e-12);
        assert!(
            generalized_eigenvalues(0, &[], &[])
                .expect("empty")
                .is_empty()
        );
    }

    #[test]
    fn finds_a_resonators_conjugate_pair() {
        // A series resistor, inductor and capacitor driven by a source: the nodes in, a and
        // out, then the source's current and the inductor's, as a circuit's equations hold
        // them, against minus the capacitances.
        let g = [
            0.01, -0.01, 0.0, 1.0, 0.0, -0.01, 0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0,
        ];
        let mut c = [0.0; 25];
        c[12] = -1e-9;
        c[24] = 1e-3;
        let found = finite(&generalized_eigenvalues(5, &g, &c).expect("converges"));
        let (damping, ringing) = (-5e4, (1e12 - 2.5e9_f64).sqrt());
        assert_eq!(found.len(), 2, "{found:?}");
        for expected in [
            Complex64::new(damping, -ringing),
            Complex64::new(damping, ringing),
        ] {
            assert!(
                found
                    .iter()
                    .any(|root| (root - expected).norm() < 1e-9 * ringing),
                "{found:?}"
            );
        }
    }
}
