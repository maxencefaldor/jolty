//! The generalized eigenvalues of dense matrix pencils, by the QZ algorithm.

use dyn_stack::{MemBuffer, MemStack};
use faer::diag::Diag;
use faer::linalg::evd::ComputeEigenvectors;
use faer::linalg::gevd::{GevdParams, gevd_cplx, gevd_scratch};
use faer::{Auto, Mat, Par, Spec};
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
/// An entry below `ε` times the largest of its matrix is taken as zero.
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
    let (a, b) = (flushed(a), flushed(b));
    // faer 0.24's QZ iteration stops after 30 sweeps per row without saying so, and leaves
    // the eigenvalues it has not found where they were: begun from NaN, they tell. The
    // pencil reversed, `(P a P, P b P)` with `P` the exchange matrix, has the same
    // eigenvalues, and its iteration takes another path, which converges where the first
    // did not on pencils met so far.
    if let Some(found) = complex_qz(n, |row, column| (a[column * n + row], b[column * n + row]))? {
        return Ok(found);
    }
    let last = n - 1;
    complex_qz(n, |row, column| {
        let index = (last - column) * n + (last - row);
        (a[index], b[index])
    })?
    .ok_or(PencilError::Unconverged)
}

/// `matrix` with its entries below `ε` times its largest made zero.
///
/// The QZ iteration's own tests take an entry that small as zero, as it is within the
/// iteration's rounding. Left in, it breeds products that underflow, and
/// faer 0.24's rotations, which divide by the magnitude of the pair they rotate, turn a
/// pair of subnormal entries into NaN, which spreads through both matrices.
fn flushed(matrix: &[f64]) -> Vec<f64> {
    let largest = matrix
        .iter()
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let least = f64::EPSILON * largest;
    matrix
        .iter()
        .map(|&value| if value.abs() < least { 0.0 } else { value })
        .collect()
}

/// The eigenvalues of the pencil whose entries `entry` gives, as `(a, b)` at a row and a
/// column, by the complex QZ iteration, which gives each eigenvalue of a conjugate pair its
/// own numerator and denominator; `None` when the iteration stopped before finding them
/// all.
fn complex_qz(
    n: usize,
    entry: impl Fn(usize, usize) -> (f64, f64),
) -> Result<Option<Vec<Eigenvalue>>, PencilError> {
    let mut a = Mat::from_fn(n, n, |row, column| {
        Complex64::new(entry(row, column).0, 0.0)
    });
    let mut b = Mat::from_fn(n, n, |row, column| {
        Complex64::new(entry(row, column).1, 0.0)
    });
    let unfound = Complex64::new(f64::NAN, f64::NAN);
    let (mut alpha, mut beta) = (Diag::zeros(n), Diag::zeros(n));
    for index in 0..n {
        alpha[index] = unfound;
        beta[index] = unfound;
    }
    // faer 0.24's blocked reduction to Hessenberg-triangular form returns zeros past its
    // first block of 32 columns, and its blocked QZ sweeps stall on pencils of 75 to about
    // 150 rows; the unblocked reduction and sweeps are exact, and as fast to 500 rows.
    let mut params = <GevdParams as Auto<Complex64>>::auto();
    params.hessenberg.block_size = 1;
    params.schur.blocking_threshold = usize::MAX;
    let spec = Spec::new(params);
    let scratch = gevd_scratch::<Complex64>(
        n,
        ComputeEigenvectors::No,
        ComputeEigenvectors::No,
        Par::Seq,
        spec,
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
        spec,
    )
    .map_err(|_| PencilError::Unconverged)?;
    let found: Vec<Eigenvalue> = (0..n)
        .map(|index| Eigenvalue {
            alpha: alpha[index],
            beta: beta[index],
        })
        .collect();
    Ok(found
        .iter()
        .all(|eigenvalue| !eigenvalue.alpha.is_nan() && !eigenvalue.beta.is_nan())
        .then_some(found))
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
    fn finds_every_pole_of_a_long_rc_ladder() {
        // A source drives 80 stages of a 1 kΩ resistor into 1 mF to the ground: its poles
        // are −(2/RC)(1 − cos((2k − 1)π / 161)), for k from 1 to 80, against the pencil of
        // the conductances and minus the capacitances, the source's node first and its
        // current last.
        let stages = 80;
        let size = stages + 2;
        let (conductance, capacitance) = (1e-3, 1e-3);
        let mut a = vec![0.0; size * size];
        let mut b = vec![0.0; size * size];
        for stage in 0..stages {
            for (row, column, sign) in [(0, 0, 1.0), (1, 1, 1.0), (0, 1, -1.0), (1, 0, -1.0)] {
                a[(stage + column) * size + stage + row] += sign * conductance;
            }
            b[(stage + 1) * size + stage + 1] = -capacitance;
        }
        a[(size - 1) * size] = 1.0;
        a[size - 1] = 1.0;
        let found = finite(&generalized_eigenvalues(size, &a, &b).expect("converges"));
        assert_eq!(found.len(), stages, "{found:?}");
        let mut poles: Vec<f64> = found.iter().map(|pole| pole.re).collect();
        poles.sort_by(f64::total_cmp);
        let mut exact: Vec<f64> = (1..=stages)
            .map(|k| {
                let angle = f64::from(u32::try_from(2 * k - 1).expect("small"))
                    * std::f64::consts::PI
                    / f64::from(u32::try_from(2 * stages + 1).expect("small"));
                -2.0 * conductance / capacitance * (1.0 - angle.cos())
            })
            .collect();
        exact.sort_by(f64::total_cmp);
        for (pole, exact) in poles.iter().zip(&exact) {
            assert!(
                (pole - exact).abs() < 1e-9 * exact.abs(),
                "{pole} against {exact}"
            );
        }
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

    #[test]
    fn takes_entries_far_below_the_largest_as_zero() {
        // a = diag(1, 2, 3) but for four subnormal entries, which faer 0.24 rotates in
        // pairs into NaN, the pencil reversed as well; b = I.
        let tiny = 1e-310;
        let a = [1.0, tiny, tiny, 0.0, 2.0, 0.0, tiny, tiny, 3.0];
        let b = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let found = finite(&generalized_eigenvalues(3, &a, &b).expect("converges"));
        let mut eigenvalues: Vec<f64> = found.iter().map(|eigenvalue| eigenvalue.re).collect();
        eigenvalues.sort_by(f64::total_cmp);
        assert_eq!(eigenvalues, [1.0, 2.0, 3.0], "{found:?}");
    }
}
