//! Dense linear solves, for the small systems whose every entry is filled.

use faer::MatRef;
use faer::linalg::solvers::Svd;
use faer::linalg::svd::SvdError;

use crate::LinearError;
use crate::lu::finite;

/// Solves the dense square system `A x = b`, for the matrix `a` of `rhs.len()` rows in
/// column-major order and `b` in `rhs`, by the least squares among the `x` in the range of
/// `A` it keeps, the least of them in norm; leaves `x` in `rhs` and returns the rank of the
/// part of `A` it keeps.
///
/// Where `A` is regular, `x = A⁻¹ b`. Where it is singular, the system has many solutions
/// or none, and `x` is orthogonal to every `u` with `uᵀ A = 0`, which no solution's
/// residual depends on: where `A` is `Φ − I` for a map with Jacobian `Φ`, each such `uᵀ x`
/// is a quantity the map conserves, which a step by `x` leaves as it was. Where `A` maps
/// part of its range onto its null space, the least-squares solution in its range is not
/// unique, and `x` is the least in norm.
///
/// With the singular value decomposition `A = U Σ Vᵀ`, a singular value at most `n ε` times
/// the largest, within the rounding of the matrix, counts as zero; so does each least one
/// in turn, as the discrepancy principle leaves out the directions a right-hand side known
/// to within `slack` cannot resolve, while it is at most `cutoff` and the part of `b` along
/// its direction and those left out before stays within `slack` in norm. With `r` singular
/// values kept, `x = U_r y`, where `y` is the least-squares solution of least norm of
/// `Σ_r V_rᵀ U_r y = U_rᵀ b`; a regular `A` has `x = V Σ⁻¹ Uᵀ b`.
///
/// # Errors
///
/// Returns [`LinearError::Singular`] when no finite solution is found: naming the columns
/// whose values in the solution are not finite, or none when the matrix's singular values
/// do not converge, as a number in it that is not finite makes them.
///
/// # Panics
///
/// Panics unless `a` holds the square of `rhs.len()` values.
pub fn solve_in_range(
    a: &[f64],
    rhs: &mut [f64],
    (cutoff, slack): (f64, f64),
) -> Result<usize, LinearError> {
    let n = rhs.len();
    let Some(decomposed) = Decomposed::new(a, n)? else {
        return Ok(0);
    };
    let projected = decomposed.projected(rhs);
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
    rhs.copy_from_slice(&solution);
    Ok(rank)
}

/// Solves `A x = b` for each right-hand side `b` of `columns`, each of as many rows as `A`,
/// as [`solve_in_range`] solves one with no singular value counting as zero but those within
/// the rounding, the matrix decomposed once for all of them; leaves each `x` in its column
/// and returns the rank of `A`.
///
/// # Errors
///
/// Returns [`LinearError::Singular`] as [`solve_in_range`] does, for any column.
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
        let solution = kept.solve(&decomposed.projected(column));
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

/// What solving in the range of a decomposed matrix keeps of it: its first `rank` singular
/// values, and where it keeps fewer than all, the decomposition of the reduced system
/// `Σ_r V_rᵀ U_r`.
struct Kept<'a> {
    decomposed: &'a Decomposed,
    rank: usize,
    reduced: Option<(Svd<f64>, Vec<f64>)>,
}

/// The error of a decomposition whose singular values did not converge.
fn unconverged(_: SvdError) -> LinearError {
    LinearError::Singular {
        columns: Vec::new(),
    }
}

impl Decomposed {
    /// The decomposition of the matrix `a` of `n` rows in column-major order; none when it
    /// has none.
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

    /// The singular values above the rounding.
    fn rounded_rank(&self) -> usize {
        self.values
            .iter()
            .take_while(|&&value| value > self.rounding)
            .count()
    }

    /// `Uᵀ b`, the parts of `b` along each column of `U`.
    fn projected(&self, rhs: &[f64]) -> Vec<f64> {
        let u = self.svd.U();
        (0..rhs.len()).map(|k| dot(u.col(k), rhs)).collect()
    }

    /// What solving in the range keeps of the first `rank` singular values.
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
    /// The solution in the range kept, of the least norm among the least-squares ones, for
    /// a right-hand side of parts `projected` along `U`'s columns: `x = U_r y`, where `y`
    /// solves `Σ_r V_rᵀ U_r y = U_rᵀ b`, or `x = V Σ⁻¹ Uᵀ b` where all are kept.
    fn solve(&self, projected: &[f64]) -> Vec<f64> {
        let decomposed = self.decomposed;
        let (u, v) = (decomposed.svd.U(), decomposed.svd.V());
        let rounding = decomposed.rounding;
        let Some((inner, values)) = &self.reduced else {
            return truncated((v, &decomposed.values, rounding), projected);
        };
        let within = truncated(
            (inner.V(), values, rounding),
            &inner_projected(inner, projected, self.rank),
        );
        (0..u.nrows())
            .map(|row| (0..self.rank).map(|k| u[(row, k)] * within[k]).sum::<f64>())
            .collect()
    }
}

/// The parts along the reduced system's left singular vectors of the first `rank` of
/// `projected`.
fn inner_projected(inner: &Svd<f64>, projected: &[f64], rank: usize) -> Vec<f64> {
    let u = inner.U();
    (0..rank)
        .map(|k| dot(u.col(k), &projected[..rank]))
        .collect()
}

/// `V Σ⁺ c` for the parts `c = Uᵀ b` of a right-hand side along the left singular vectors
/// of the decomposition `(U, Σ, V)`, `(V, Σ)` given, a singular value at most `cutoff`
/// counting as zero.
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

/// The dot product of a column and a vector of its length.
fn dot(column: faer::ColRef<'_, f64>, vector: &[f64]) -> f64 {
    column.iter().zip(vector).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No singular value counts as zero but those within the rounding: exact zero
    /// `(relative, absolute)` threshold tuple for unit tests of `solve_in_range`.
    const EXACT: (f64, f64) = (0.0, 0.0);

    /// `A x` for the dense matrix `a` of `x.len()` rows, in column-major order.
    fn product(a: &[f64], x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..n)
            .map(|row| (0..n).map(|column| a[column * n + row] * x[column]).sum())
            .collect()
    }

    #[test]
    fn solves_a_regular_system() {
        // [[2, 1], [1, 3]] x = [3, 5]: x = [0.8, 1.4].
        let mut rhs = [3.0, 5.0];
        assert_eq!(
            solve_in_range(&[2.0, 1.0, 1.0, 3.0], &mut rhs, EXACT),
            Ok(2)
        );
        assert!((rhs[0] - 0.8).abs() < 1e-15 && (rhs[1] - 1.4).abs() < 1e-15);
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
        let mut rhs = product(&a, &expected);
        assert_eq!(solve_in_range(&a, &mut rhs, EXACT), Ok(n));
        for (found, expected) in rhs.iter().zip(&expected) {
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
        let mut rhs = [1.0, 0.0];
        assert_eq!(
            solve_in_range(&[-0.5, 0.0, 0.2, 0.0], &mut rhs, EXACT),
            Ok(1)
        );
        assert!(
            (rhs[0] + 2.0).abs() < 1e-15 && rhs[1].abs() < 1e-15,
            "{rhs:?}"
        );
        // So does an inconsistent system, whose second row asks for a change of x₂ that no
        // x gives: the least-squares solution solves the first row alone.
        let mut rhs = [1.0, 1e-3];
        assert_eq!(
            solve_in_range(&[-0.5, 0.0, 0.2, 0.0], &mut rhs, EXACT),
            Ok(1)
        );
        assert!(
            (rhs[0] + 2.0).abs() < 1e-15 && rhs[1].abs() < 1e-15,
            "{rhs:?}"
        );
    }

    #[test]
    fn keeps_the_conserved_quantity_of_a_singular_map_in_any_coordinates() {
        // The same map in the coordinates x' = M x, M = [[1, 0], [1, 1]]: M (Φ − I) M⁻¹ =
        // [[−0.7, 0.2], [−0.7, 0.2]], and M [1, 0] = [1, 1]. The conserved quantity is now
        // x₂' − x₁', and the solution M [−2, 0] = [−2, −2] keeps it, where the least-norm
        // solution, along [−0.7, 0.2], would not.
        let mut rhs = [1.0, 1.0];
        assert_eq!(
            solve_in_range(&[-0.7, -0.7, 0.2, 0.2], &mut rhs, EXACT),
            Ok(1)
        );
        assert!(
            (rhs[0] + 2.0).abs() < 1e-14 && (rhs[1] + 2.0).abs() < 1e-14,
            "{rhs:?}"
        );
    }

    #[test]
    fn a_map_that_carries_its_range_into_its_null_space_takes_the_least_step() {
        // [[0, 1], [0, 0]]: the range, the first axis, is the null space too, so no x in the
        // range changes the residual, and the least of them is zero.
        let mut rhs = [1.0, 0.0];
        assert_eq!(
            solve_in_range(&[0.0, 0.0, 1.0, 0.0], &mut rhs, EXACT),
            Ok(1)
        );
        assert!(rhs.iter().all(|value| value.abs() < 1e-15), "{rhs:?}");
    }

    #[test]
    fn a_singular_value_within_the_rounding_counts_as_zero() {
        // The second row is the first, scaled, short of it by a rounding: rank 1.
        let scale = 3.0;
        let mut rhs = [1.0, scale];
        let a = [1.0, scale * (1.0 + f64::EPSILON), 2.0, scale * 2.0];
        assert_eq!(solve_in_range(&a, &mut rhs, EXACT), Ok(1));
        assert!(
            rhs.iter()
                .all(|value| value.is_finite() && value.abs() < 1.0),
            "{rhs:?}"
        );
    }

    #[test]
    fn a_direction_the_right_hand_side_cannot_resolve_is_left_out() {
        // diag(1, 1e-4): the second direction is resolved to 1e4 times the right-hand
        // side's error. Known to within 1, a part of 0.5 along it is left out; a part of 2
        // is not, nor is any part along a singular value above the cutoff.
        let a = [1.0, 0.0, 0.0, 1e-4];
        let mut rhs = [3.0, 0.5];
        assert_eq!(solve_in_range(&a, &mut rhs, (1e-3, 1.0)), Ok(1));
        assert!(
            (rhs[0] - 3.0).abs() < 1e-15 && rhs[1].abs() < 1e-15,
            "{rhs:?}"
        );
        let mut rhs = [3.0, 2.0];
        assert_eq!(solve_in_range(&a, &mut rhs, (1e-3, 1.0)), Ok(2));
        assert!((rhs[1] - 2e4).abs() < 1e-9, "{rhs:?}");
        let mut rhs = [3.0, 0.5];
        assert_eq!(solve_in_range(&a, &mut rhs, (1e-5, 1.0)), Ok(2));
        // The parts left out add up: with a third direction of 2e-4 and 0.9 along it, the
        // two together pass 1, so only the second, the least, is left out.
        let a = [1.0, 0.0, 0.0, 0.0, 1e-4, 0.0, 0.0, 0.0, 2e-4];
        let mut rhs = [3.0, 0.5, 0.9];
        assert_eq!(solve_in_range(&a, &mut rhs, (1e-3, 1.0)), Ok(2));
        assert!(
            rhs[1].abs() < 1e-15 && (rhs[2] - 4.5e3).abs() < 1e-9,
            "{rhs:?}"
        );
    }

    #[test]
    fn a_number_that_is_not_finite_has_no_solution() {
        // The projections onto the decomposition's directions carry the NaN into every
        // value of the solution, each of which the error names.
        let mut rhs = [1.0, f64::NAN];
        assert_eq!(
            solve_in_range(&[1.0, 0.0, 0.0, 1.0], &mut rhs, EXACT),
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
