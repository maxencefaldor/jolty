//! Dense linear solves, for the small systems whose every entry is filled.

use faer::MatRef;
use faer::linalg::solvers::Svd;
use faer::linalg::svd::SvdError;

use crate::LinearError;
use crate::lu::finite;

/// Solves the dense square system `A x = b` in the range of `A`, writing `x` into
/// `right_hand_side` and returning the numerical rank retained.
///
/// Matrix `a` holds `right_hand_side.len()` rows in column-major order. When `A` is
/// nonsingular, `x = A⁻¹ b`. When `A` is singular, `x` is the minimum-norm least-squares
/// solution orthogonal to every left null vector `u` (`uᵀ A = 0`), preserving invariants
/// conserved by a map `Φ` when `A = Φ − I`.
///
/// Using the SVD `A = U Σ Vᵀ`, singular values at most `n ε σ_max` are treated as zero.
/// Additional trailing singular values at most `cutoff` are dropped under the discrepancy
/// principle while the omitted projection norm of `b` stays within `slack`. With `r`
/// singular values retained, `x = U_r y` where `y` is the minimum-norm least-squares
/// solution of `Σ_r V_rᵀ U_r y = U_rᵀ b`.
///
/// # Errors
///
/// Returns [`LinearError::Singular`] when no finite solution exists, listing columns with
/// non-finite solution entries (or empty columns when the SVD fails to converge).
///
/// # Panics
///
/// Panics unless `a` holds `right_hand_side.len().pow(2)` values.
pub fn solve_in_range(
    a: &[f64],
    right_hand_side: &mut [f64],
    (cutoff, slack): (f64, f64),
) -> Result<usize, LinearError> {
    let n = right_hand_side.len();
    let Some(decomposed) = Decomposed::new(a, n)? else {
        return Ok(0);
    };
    let projected = project(decomposed.svd.U(), right_hand_side);
    let values = &decomposed.values;
    let mut rank = decomposed.rounded_rank();
    let mut left_out: f64 = projected[rank..].iter().map(|part| part * part).sum();
    while let Some(least) = rank.checked_sub(1)
        && values[least] <= cutoff
        && left_out + projected[least] * projected[least] <= slack * slack
    {
        left_out += projected[least] * projected[least];
        rank = least;
    }
    let solution = decomposed.kept(rank)?.solve(&projected);
    finite(&solution)?;
    right_hand_side.copy_from_slice(&solution);
    Ok(rank)
}

/// Solves `A x = b` in the range of `A` for each right-hand side in `columns`, reusing one
/// SVD across all columns, and returns the numerical rank of `A`.
///
/// Each column is overwritten in place with its solution, treating only singular values within
/// rounding precision as zero (matching [`solve_in_range`] with zero discrepancy thresholds).
///
/// # Errors
///
/// Returns [`LinearError::Singular`] if the SVD does not converge or any column yields a
/// non-finite solution.
///
/// # Panics
///
/// Panics if the columns differ in length, or if `columns` is non-empty and `a` does not hold
/// the square of their length.
pub fn solve_each_in_range(a: &[f64], columns: &mut [Vec<f64>]) -> Result<usize, LinearError> {
    let Some(first) = columns.first() else {
        return Ok(0);
    };
    let n = first.len();
    assert!(
        columns.iter().all(|column| column.len() == n),
        "every column has the same length"
    );
    let Some(decomposed) = Decomposed::new(a, n)? else {
        return Ok(0);
    };
    let rank = decomposed.rounded_rank();
    let kept = decomposed.kept(rank)?;
    for column in columns {
        let solution = kept.solve(&project(decomposed.svd.U(), column));
        finite(&solution)?;
        column.copy_from_slice(&solution);
    }
    Ok(rank)
}

/// The singular value decomposition `A = U Σ Vᵀ` of a square matrix, and the rounding below
/// which its singular values count as zero: `n ε` times the largest.
struct Decomposed {
    svd: Svd<f64>,
    values: Vec<f64>,
    rounding: f64,
}

/// Truncated SVD state retaining the first `rank` singular values and (when rank-deficient)
/// the SVD of the reduced operator `Σ_r V_rᵀ U_r`.
struct Kept<'a> {
    decomposed: &'a Decomposed,
    rank: usize,
    reduced: Option<(Svd<f64>, Vec<f64>)>,
}

/// Converts an unconverged SVD error into [`LinearError::Singular`].
fn unconverged(_: SvdError) -> LinearError {
    LinearError::Singular {
        columns: Vec::new(),
    }
}

impl Decomposed {
    /// Computes the SVD of column-major `n × n` matrix `a`, returning `None` when `n == 0`.
    fn new(a: &[f64], n: usize) -> Result<Option<Self>, LinearError> {
        assert_eq!(
            a.len(),
            n * n,
            "a square matrix of the right-hand side's rows"
        );
        if n == 0 {
            return Ok(None);
        }
        let svd = MatRef::from_column_major_slice(a, n, n)
            .svd()
            .map_err(unconverged)?;
        let values: Vec<f64> = svd.S().column_vector().iter().copied().collect();
        let rounding = f64::from(u32::try_from(n).unwrap_or(u32::MAX)) * f64::EPSILON * values[0];
        Ok(Some(Self {
            svd,
            values,
            rounding,
        }))
    }

    /// Counts singular values strictly above the rounding threshold.
    fn rounded_rank(&self) -> usize {
        self.values
            .iter()
            .take_while(|&&value| value > self.rounding)
            .count()
    }

    /// Prepares the reduced range-space solver for the leading `rank` singular values.
    fn kept(&self, rank: usize) -> Result<Kept<'_>, LinearError> {
        let reduced = if rank == self.values.len() {
            None
        } else {
            let (u, v) = (self.svd.U(), self.svd.V());
            let mut reduced = v.subcols(0, rank).transpose() * u.subcols(0, rank);
            for (row, &value) in self.values[..rank].iter().enumerate() {
                for column in 0..rank {
                    reduced[(row, column)] *= value;
                }
            }
            let inner = reduced.svd().map_err(unconverged)?;
            let values = inner.S().column_vector().iter().copied().collect();
            Some((inner, values))
        };
        Ok(Kept {
            decomposed: self,
            rank,
            reduced,
        })
    }
}

impl Kept<'_> {
    /// Computes the minimum-norm least-squares solution in the retained range from
    /// `projected = Uᵀ b` (`x = U_r y` for rank-deficient systems or `x = V Σ⁻¹ Uᵀ b` at
    /// full rank).
    fn solve(&self, projected: &[f64]) -> Vec<f64> {
        let decomposed = self.decomposed;
        let (u, v) = (decomposed.svd.U(), decomposed.svd.V());
        let rounding = decomposed.rounding;
        let Some((inner, values)) = &self.reduced else {
            return truncated((v, &decomposed.values, rounding), projected);
        };
        let within = truncated(
            (inner.V(), values, rounding),
            &project(inner.U(), &projected[..self.rank]),
        );
        (0..u.nrows())
            .map(|row| (0..self.rank).map(|k| u[(row, k)] * within[k]).sum::<f64>())
            .collect()
    }
}

/// Computes `uᵀ vector`, taking the dot product of each leading column of `u` with `vector`.
fn project(u: MatRef<'_, f64>, vector: &[f64]) -> Vec<f64> {
    (0..vector.len())
        .map(|column| u.col(column).iter().zip(vector).map(|(x, y)| x * y).sum())
        .collect()
}

/// Computes `V Σ⁺ projected` from `(v, values)` and `projected = Uᵀ b`, zeroing singular
/// values at or below `cutoff`.
fn truncated((v, values, cutoff): (MatRef<'_, f64>, &[f64], f64), projected: &[f64]) -> Vec<f64> {
    let mut solution = vec![0.0; v.nrows()];
    for (k, (&value, &part)) in values.iter().zip(projected).enumerate() {
        if value <= cutoff {
            break;
        }
        let along = part / value;
        for (entry, &direction) in solution.iter_mut().zip(v.col(k).iter()) {
            *entry += along * direction;
        }
    }
    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero `(cutoff, slack)` threshold tuple disabling extra singular-value dropping beyond
    /// rounding precision in unit tests of `solve_in_range`.
    const EXACT: (f64, f64) = (0.0, 0.0);

    /// Computes `A x` for column-major dense matrix `a` of dimension `x.len()`.
    fn product(a: &[f64], x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|row| (0..n).map(|column| a[column * n + row] * x[column]).sum())
            .collect()
    }

    #[test]
    fn solves_a_regular_system() {
        // [[2, 1], [1, 3]] x = [3, 5]: x = [0.8, 1.4].
        let mut right_hand_side = [3.0, 5.0];
        assert_eq!(
            solve_in_range(&[2.0, 1.0, 1.0, 3.0], &mut right_hand_side, EXACT),
            Ok(2)
        );
        assert!(
            (right_hand_side[0] - 0.8).abs() < 1e-15 && (right_hand_side[1] - 1.4).abs() < 1e-15
        );
    }

    #[test]
    fn solves_a_large_regular_system_as_its_inverse_does() {
        // A matrix of 80 rows, past the blocks dense factorizations work in, that is
        // diagonally dominant, so regular: A x = A [1, 2, …] gives back [1, 2, …].
        let n = 80;
        let a: Vec<f64> = (0..n * n)
            .map(|index| {
                let (row, column) = (index % n, index / n);
                let wave = f64::from(u32::try_from(index).unwrap()).sin();
                if row == column {
                    f64::from(u32::try_from(n).unwrap()) + wave
                } else {
                    wave
                }
            })
            .collect();
        let expected: Vec<f64> = (1..=n)
            .map(|k| f64::from(u32::try_from(k).unwrap()))
            .collect();
        let mut right_hand_side = product(&a, &expected);
        assert_eq!(solve_in_range(&a, &mut right_hand_side, EXACT), Ok(n));
        for (found, expected) in right_hand_side.iter().zip(&expected) {
            assert!(
                (found - expected).abs() < 1e-11 * expected,
                "{found} for {expected}"
            );
        }
    }

    #[test]
    fn leaves_alone_what_a_singular_map_conserves() {
        // Φ = [[0.5, 0.2], [0, 1]]: the second component repeats whatever the first, so
        // Φ − I = [[−0.5, 0.2], [0, 0]] is singular, and every x with x₂ fixed and
        // −0.5 x₁ + 0.2 x₂ = 1 solves it. The step in its range keeps x₂ as it is, where
        // the least-norm solution would move it.
        let mut right_hand_side = [1.0, 0.0];
        assert_eq!(
            solve_in_range(&[-0.5, 0.0, 0.2, 0.0], &mut right_hand_side, EXACT),
            Ok(1)
        );
        assert!(
            (right_hand_side[0] + 2.0).abs() < 1e-15 && right_hand_side[1].abs() < 1e-15,
            "{right_hand_side:?}"
        );
        // An inconsistent system whose second row demands an unattainable change in x₂ also
        // keeps x₂ fixed; the least-squares solution satisfies the first row alone.
        let mut right_hand_side = [1.0, 1e-3];
        assert_eq!(
            solve_in_range(&[-0.5, 0.0, 0.2, 0.0], &mut right_hand_side, EXACT),
            Ok(1)
        );
        assert!(
            (right_hand_side[0] + 2.0).abs() < 1e-15 && right_hand_side[1].abs() < 1e-15,
            "{right_hand_side:?}"
        );
    }

    #[test]
    fn keeps_the_conserved_quantity_of_a_singular_map_in_any_coordinates() {
        // The same map in the coordinates x' = M x, M = [[1, 0], [1, 1]]: M (Φ − I) M⁻¹ =
        // [[−0.7, 0.2], [−0.7, 0.2]], and M [1, 0] = [1, 1]. The conserved quantity is now
        // x₂' − x₁', and the solution M [−2, 0] = [−2, −2] keeps it, where the least-norm
        // solution, along [−0.7, 0.2], would not.
        let mut right_hand_side = [1.0, 1.0];
        assert_eq!(
            solve_in_range(&[-0.7, -0.7, 0.2, 0.2], &mut right_hand_side, EXACT),
            Ok(1)
        );
        assert!(
            (right_hand_side[0] + 2.0).abs() < 1e-14 && (right_hand_side[1] + 2.0).abs() < 1e-14,
            "{right_hand_side:?}"
        );
    }

    #[test]
    fn a_map_that_carries_its_range_into_its_null_space_takes_the_least_step() {
        // [[0, 1], [0, 0]]: the range, the first axis, is the null space too, so no x in the
        // range changes the residual, and the least of them is zero.
        let mut right_hand_side = [1.0, 0.0];
        assert_eq!(
            solve_in_range(&[0.0, 0.0, 1.0, 0.0], &mut right_hand_side, EXACT),
            Ok(1)
        );
        assert!(
            right_hand_side.iter().all(|value| value.abs() < 1e-15),
            "{right_hand_side:?}"
        );
    }

    #[test]
    fn a_singular_value_within_the_rounding_counts_as_zero() {
        // The second row is the first, scaled, short of it by a rounding: rank 1.
        let scale = 3.0;
        let mut right_hand_side = [1.0, scale];
        let a = [1.0, scale * (1.0 + f64::EPSILON), 2.0, scale * 2.0];
        assert_eq!(solve_in_range(&a, &mut right_hand_side, EXACT), Ok(1));
        assert!(
            right_hand_side
                .iter()
                .all(|value| value.is_finite() && value.abs() < 1.0),
            "{right_hand_side:?}"
        );
    }

    #[test]
    fn a_direction_the_right_hand_side_cannot_resolve_is_left_out() {
        // diag(1, 1e-4): the second direction amplifies right-hand-side error by 1e4. With
        // slack 1, a component of 0.5 is omitted; a component of 2 or a singular value above
        // cutoff is retained.
        let a = [1.0, 0.0, 0.0, 1e-4];
        let mut right_hand_side = [3.0, 0.5];
        assert_eq!(solve_in_range(&a, &mut right_hand_side, (1e-3, 1.0)), Ok(1));
        assert!(
            (right_hand_side[0] - 3.0).abs() < 1e-15 && right_hand_side[1].abs() < 1e-15,
            "{right_hand_side:?}"
        );
        let mut right_hand_side = [3.0, 2.0];
        assert_eq!(solve_in_range(&a, &mut right_hand_side, (1e-3, 1.0)), Ok(2));
        assert!(
            (right_hand_side[1] - 2e4).abs() < 1e-9,
            "{right_hand_side:?}"
        );
        let mut right_hand_side = [3.0, 0.5];
        assert_eq!(solve_in_range(&a, &mut right_hand_side, (1e-5, 1.0)), Ok(2));
        // Omitted projections accumulate in norm: with a third direction of 2e-4 and 0.9
        // along it, omitting both exceeds slack 1, so only the smallest direction is omitted.
        let a = [1.0, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 2e-4];
        let mut right_hand_side = [3.0, 0.5, 0.9];
        assert_eq!(solve_in_range(&a, &mut right_hand_side, (1e-3, 1.0)), Ok(2));
        assert!(
            right_hand_side[1].abs() < 1e-15 && (right_hand_side[2] - 4.5e3).abs() < 1e-9,
            "{right_hand_side:?}"
        );
    }

    #[test]
    fn a_number_that_is_not_finite_has_no_solution() {
        // The projections onto the decomposition's directions carry the NaN into every
        // value of the solution, each of which the error names.
        let mut right_hand_side = [1.0, f64::NAN];
        assert_eq!(
            solve_in_range(&[1.0, 0.0, 0.0, 1.0], &mut right_hand_side, EXACT),
            Err(LinearError::Singular {
                columns: vec![0, 1]
            })
        );
    }

    #[test]
    fn solves_each_column_as_one_solve_does() {
        // The regular system above, and the singular one, whose columns' solutions keep
        // x₂ where it is: each column as `solve_in_range` solves it alone.
        for a in [[2.0, 1.0, 1.0, 3.0], [-0.5, 0.0, 0.2, 0.0]] {
            let mut columns = vec![vec![3.0, 5.0], vec![1.0, 0.0], vec![1.0, 1e-3]];
            let alone: Vec<Vec<f64>> = columns
                .iter()
                .map(|column| {
                    let mut solved = column.clone();
                    solve_in_range(&a, &mut solved, EXACT).unwrap();
                    solved
                })
                .collect();
            let rank = solve_each_in_range(&a, &mut columns).unwrap();
            assert_eq!(rank, if a[1] == 0.0 { 1 } else { 2 });
            assert_eq!(columns, alone);
        }
    }

    #[test]
    fn solve_each_in_range_handles_empty_columns_without_panicking() {
        assert_eq!(solve_each_in_range(&[2.0, 1.0, 1.0, 3.0], &mut []), Ok(0));
        assert_eq!(solve_each_in_range(&[], &mut []), Ok(0));
    }

    #[test]
    #[should_panic(expected = "every column has the same length")]
    fn solve_each_in_range_rejects_mismatched_column_lengths() {
        let mut columns = vec![vec![1.0, 2.0], vec![3.0]];
        let _ = solve_each_in_range(&[1.0, 0.0, 0.0, 1.0], &mut columns);
    }
}
