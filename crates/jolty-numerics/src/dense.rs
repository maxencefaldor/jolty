//! Dense linear solves, for the small systems whose every entry is filled.

use faer::Mat;
use faer::linalg::solvers::Solve;

use crate::LinearError;

/// Solves `A x = rhs` for the dense square matrix `a` of `rhs.len()` rows, in column-major
/// order, by LU factorization with partial pivoting, and leaves `x` in `rhs`.
///
/// # Errors
///
/// Returns [`LinearError::Singular`] when the solution is not finite, which a singular
/// matrix causes.
///
/// # Panics
///
/// Panics unless `a` holds the square of `rhs.len()` values.
pub fn solve_dense(a: &[f64], rhs: &mut [f64]) -> Result<(), LinearError> {
    let n = rhs.len();
    assert_eq!(
        a.len(),
        n * n,
        "a square matrix of the right-hand side's rows"
    );
    if n == 0 {
        return Ok(());
    }
    let matrix = Mat::from_fn(n, n, |row, column| a[column * n + row]);
    let solution = matrix
        .partial_piv_lu()
        .solve(Mat::from_fn(n, 1, |row, _| rhs[row]));
    for (row, value) in rhs.iter_mut().enumerate() {
        *value = solution[(row, 0)];
    }
    if rhs.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(LinearError::Singular)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solves_a_small_system() {
        // [[2, 1], [1, 3]] x = [3, 5]: x = [0.8, 1.4].
        let mut rhs = [3.0, 5.0];
        solve_dense(&[2.0, 1.0, 1.0, 3.0], &mut rhs).expect("regular");
        assert!((rhs[0] - 0.8).abs() < 1e-15 && (rhs[1] - 1.4).abs() < 1e-15);
        let mut singular = [1.0, 1.0];
        assert_eq!(
            solve_dense(&[1.0, 1.0, 1.0, 1.0], &mut singular),
            Err(LinearError::Singular)
        );
    }
}
