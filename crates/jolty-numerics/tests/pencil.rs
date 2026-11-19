//! The QZ iteration on pencils whose eigenvalues are known.
#![cfg(test)]

use jolty_numerics::{Complex64, Eigenvalue, generalized_eigenvalues};
use proptest::prelude::*;

/// Returns the finite generalized eigenvalues from `found` whose denominator is non-negligible
/// relative to their numerator.
fn finite(found: &[Eigenvalue]) -> Vec<Complex64> {
    found
        .iter()
        .filter(|eigenvalue| eigenvalue.beta.norm() > 1e-12 * eigenvalue.alpha.norm())
        .map(|eigenvalue| eigenvalue.alpha / eigenvalue.beta)
        .collect()
}

/// Asserts that each value in `expected` matches a distinct value in `found` within relative
/// `tolerance`, greedily pairing nearest values first.
fn assert_match(found: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(
        found.len(),
        expected.len(),
        "{found:?} against {expected:?}"
    );
    let mut unmatched = found.to_vec();
    for value in expected {
        let nearest = unmatched
            .iter()
            .enumerate()
            .min_by(|(_, x), (_, y)| (*x - value).norm().total_cmp(&(*y - value).norm()))
            .map(|(index, _)| index)
            .expect("as many found as expected");
        let candidate = unmatched.swap_remove(nearest);
        assert!(
            (candidate - value).norm() <= tolerance * value.norm().max(1.0),
            "{candidate} against {value}"
        );
    }
}

#[test]
fn a_pencil_faer_did_not_converge_on_agrees_with_lapack() {
    // Pencil from a random 193-device circuit where faer 0.24's QZ exhausted sweeps on Apple
    // silicon. Eigenvalue condition numbers reach 4.4e10, so rounding alone can shift values
    // by up to 1e-5 relative; this QZ implementation agrees with LAPACK within 4e-8.
    let text = include_str!("pencils/stalled.txt");
    let mut lines = text.lines().filter(|line| !line.starts_with('#'));
    let n: usize = lines
        .next()
        .and_then(|line| line.parse().ok())
        .expect("the dimension");
    let (mut a, mut b) = (vec![0.0; n * n], vec![0.0; n * n]);
    let mut expected = Vec::new();
    for line in lines {
        let fields: Vec<&str> = line.split(' ').collect();
        let number = |index: usize| -> f64 { fields[index].parse().expect("a number") };
        match fields[0] {
            "eigenvalue" => expected.push(Complex64::new(number(1), number(2))),
            matrix => {
                let row: usize = fields[1].parse().expect("a row");
                let column: usize = fields[2].parse().expect("a column");
                let entries = if matrix == "a" { &mut a } else { &mut b };
                entries[column * n + row] = number(3);
            }
        }
    }
    let found = generalized_eigenvalues(n, &a, &b).expect("converges");
    assert_match(&finite(&found), &expected, 1e-5);
}

/// Constructs a column-major `n × n` real orthogonal matrix by applying Givens rotations for
/// `angles` across cycling adjacent row pairs.
fn orthogonal(n: usize, angles: &[f64]) -> Vec<f64> {
    let mut matrix = vec![0.0; n * n];
    for index in 0..n {
        matrix[index * n + index] = 1.0;
    }
    if n < 2 {
        return matrix;
    }
    for (step, angle) in angles.iter().enumerate() {
        let row = step % (n - 1);
        let (cosine, sine) = (angle.cos(), angle.sin());
        for column in 0..n {
            let (x, y) = (matrix[column * n + row], matrix[column * n + row + 1]);
            matrix[column * n + row] = cosine * x - sine * y;
            matrix[column * n + row + 1] = sine * x + cosine * y;
        }
    }
    matrix
}

/// Computes column-major `left · diag(diagonal) · right` for `n × n` factors.
fn transformed(n: usize, left: &[f64], diagonal: &[f64], right: &[f64]) -> Vec<f64> {
    let mut product = vec![0.0; n * n];
    for column in 0..n {
        for row in 0..n {
            product[column * n + row] = (0..n)
                .map(|k| left[k * n + row] * diagonal[k] * right[column * n + k])
                .sum();
        }
    }
    product
}

proptest! {
    /// Verifies that orthogonal equivalence preserves finite and infinite eigenvalues of a
    /// diagonal pencil `(diag(α), diag(β))` within `1e-9` relative tolerance.
    #[test]
    fn eigenvalues_survive_orthogonal_equivalence(
        pairs in proptest::collection::vec(
            (-1.0..1.0f64, prop_oneof![Just(0.0), 0.1..1.0f64, -1.0..-0.1f64]),
            1..14,
        ),
        angles in proptest::collection::vec(0.0..std::f64::consts::TAU, 60),
    ) {
        let n = pairs.len();
        let alphas: Vec<f64> = pairs.iter().map(|&(alpha, _)| alpha).collect();
        let betas: Vec<f64> = pairs.iter().map(|&(_, beta)| beta).collect();
        let (left, right) = (orthogonal(n, &angles[..30]), orthogonal(n, &angles[30..]));
        let a = transformed(n, &left, &alphas, &right);
        let b = transformed(n, &left, &betas, &right);
        let found = generalized_eigenvalues(n, &a, &b).expect("converges");
        prop_assert_eq!(found.len(), n);
        let infinite = found
            .iter()
            .filter(|eigenvalue| eigenvalue.beta.norm() <= 1e-12 * eigenvalue.alpha.norm())
            .count();
        let expected: Vec<Complex64> = pairs
            .iter()
            .filter(|&&(_, beta)| beta != 0.0)
            .map(|&(alpha, beta)| Complex64::new(alpha / beta, 0.0))
            .collect();
        prop_assert_eq!(infinite + expected.len(), n, "{:?}", found);
        assert_match(&finite(&found), &expected, 1e-9);
    }
}
