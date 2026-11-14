//! The sparse LU solver against a dense reference: on random sparse matrices whose large
//! entries lie off the diagonal, on the matrices of random circuits, real and complex, after
//! their values change, and on singular and nearly singular matrices.
#![cfg(test)]

use std::sync::Arc;

use jolty_numerics::{
    Complex64, Entry, LinearError, Ordering, Pattern, PatternBuilder, Scalar, SparseLu, Statistics,
};
use proptest::prelude::*;
use proptest::sample::Index;

/// A square matrix's claimed positions, each with its value, positions repeated adding up.
#[derive(Debug, Clone)]
struct Matrix<T> {
    dimension: usize,
    positions: Vec<(usize, usize)>,
    values: Vec<T>,
}

impl<T: Scalar> Matrix<T> {
    /// The pattern of the positions, and the entry of each.
    fn pattern(&self) -> (Arc<Pattern>, Vec<Entry>) {
        let mut builder = PatternBuilder::new();
        let entries = self
            .positions
            .iter()
            .map(|&(row, column)| builder.entry(row, column))
            .collect();
        (Arc::new(builder.finish(self.dimension)), entries)
    }

    /// The values by slot of `pattern`, whose `entries` the positions are.
    fn stored(&self, pattern: &Pattern, entries: &[Entry]) -> Vec<T> {
        let mut stored = vec![T::ZERO; pattern.len()];
        for (&entry, &value) in entries.iter().zip(&self.values) {
            stored[pattern.slot(entry)] += value;
        }
        stored
    }

    /// The matrix, dense by rows.
    fn dense(&self) -> Vec<Vec<T>> {
        let mut dense = vec![vec![T::ZERO; self.dimension]; self.dimension];
        for (&(row, column), &value) in self.positions.iter().zip(&self.values) {
            dense[row][column] += value;
        }
        dense
    }
}

/// The transpose of the dense `matrix`.
fn transposed<T: Scalar>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    (0..matrix.len())
        .map(|row| matrix.iter().map(|other| other[row]).collect())
        .collect()
}

/// The product of the dense `matrix` and `x`.
fn product<T: Scalar>(matrix: &[Vec<T>], x: &[T]) -> Vec<T> {
    matrix
        .iter()
        .map(|row| {
            row.iter()
                .zip(x)
                .fold(T::ZERO, |sum, (&entry, &value)| sum + entry * value)
        })
        .collect()
}

/// The largest magnitude in `values`.
fn largest<T: Scalar>(values: &[T]) -> f64 {
    values
        .iter()
        .map(|value| value.magnitude())
        .fold(0.0, f64::max)
}

/// The infinity norm of the dense `matrix`: its largest row sum of magnitudes.
fn norm<T: Scalar>(matrix: &[Vec<T>]) -> f64 {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| value.magnitude()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// The inverse of the dense `matrix` by Gauss–Jordan elimination with partial pivoting, or
/// `None` when a column has no nonzero pivot.
fn inverse<T: Scalar + One>(matrix: &[Vec<T>]) -> Option<Vec<Vec<T>>> {
    let n = matrix.len();
    let one = T::ONE;
    let mut work: Vec<Vec<T>> = matrix.to_vec();
    let mut inverse: Vec<Vec<T>> = (0..n)
        .map(|row| {
            (0..n)
                .map(|column| if row == column { one } else { T::ZERO })
                .collect()
        })
        .collect();
    for column in 0..n {
        let pivot = (column..n).max_by(|&a, &b| {
            work[a][column]
                .magnitude()
                .total_cmp(&work[b][column].magnitude())
        })?;
        if work[pivot][column].magnitude() == 0.0 {
            return None;
        }
        work.swap(column, pivot);
        inverse.swap(column, pivot);
        let scale = work[column][column];
        for value in work[column].iter_mut().chain(inverse[column].iter_mut()) {
            *value = *value / scale;
        }
        for row in 0..n {
            let factor = work[row][column];
            if row != column && factor != T::ZERO {
                for index in 0..n {
                    let (above, below) = (work[column][index], inverse[column][index]);
                    work[row][index] -= factor * above;
                    inverse[row][index] -= factor * below;
                }
            }
        }
    }
    Some(inverse)
}

/// The ones of the scalars the tests use.
trait One {
    const ONE: Self;
}

impl One for f64 {
    const ONE: Self = 1.0;
}

impl One for Complex64 {
    const ONE: Self = Complex64::new(1.0, 0.0);
}

/// A right-hand side of `dimension` rows with no zero and no two values alike.
fn rhs<T: Scalar + One>(dimension: usize) -> Vec<T> {
    (0..dimension)
        .map(|row| {
            let mut value = T::ONE;
            for _ in 0..row % 5 {
                value += T::ONE;
            }
            if row % 2 == 0 { value } else { -value }
        })
        .collect()
}

/// The relative backward error of `x` as a solution of `matrix x = b`.
fn backward_error<T: Scalar>(matrix: &[Vec<T>], x: &[T], b: &[T]) -> f64 {
    let residual: Vec<T> = product(matrix, x)
        .into_iter()
        .zip(b)
        .map(|(found, &wanted)| found - wanted)
        .collect();
    largest(&residual) / (norm(matrix) * largest(x) + largest(b))
}

/// Checks that `lu`, factoring `matrix`, solves it and its transpose as the dense reference
/// does: each solution's backward error within a few thousand roundings, and its distance
/// from the reference's within its condition number times as much.
fn check<T: Scalar + One>(lu: &mut SparseLu<T>, matrix: &Matrix<T>) -> Result<(), TestCaseError> {
    check_within(lu, matrix, 1e-12)
}

/// Checks as [`check`] does, with each solution's backward error within `backward`, and its
/// distance from the reference's within its condition number times as much.
fn check_within<T: Scalar + One>(
    lu: &mut SparseLu<T>,
    matrix: &Matrix<T>,
    backward: f64,
) -> Result<(), TestCaseError> {
    let (_, entries) = matrix.pattern();
    let dense = matrix.dense();
    let inverted = inverse(&dense).ok_or_else(|| TestCaseError::reject("singular"))?;
    let condition = norm(&dense) * norm(&inverted);
    prop_assume!(condition < 1e10, "too ill-conditioned for a sharp check");
    let stored = matrix.stored(lu.pattern(), &entries);
    let b: Vec<T> = rhs(matrix.dimension);
    let mut factors = lu.factor(&stored).map_err(|error| {
        TestCaseError::fail(format!("{error} for a matrix of condition {condition:e}"))
    })?;
    for (transpose, reference) in [(false, dense.clone()), (true, transposed(&dense))] {
        let mut x = b.clone();
        if transpose {
            factors.solve_transpose(&mut x)
        } else {
            factors.solve(&mut x)
        }
        .map_err(|error| TestCaseError::fail(error.to_string()))?;
        let error = backward_error(&reference, &x, &b);
        prop_assert!(
            error < backward,
            "backward error {error:e}, transposed: {transpose}"
        );
        let wanted = product(&inverse(&reference).expect("regular"), &b);
        let apart: Vec<T> = x.iter().zip(&wanted).map(|(&a, &b)| a - b).collect();
        let distance = largest(&apart) / largest(&wanted);
        prop_assert!(
            distance < backward * condition.max(1.0),
            "{distance:e} from the reference, of condition {condition:e}, transposed: {transpose}"
        );
    }
    Ok(())
}

/// A solver for `matrix`'s pattern, and the pattern's entries.
fn solver<T: Scalar>(matrix: &Matrix<T>) -> (SparseLu<T>, Vec<Entry>) {
    let (pattern, entries) = matrix.pattern();
    let ordering = Ordering::new(pattern).expect("fits");
    (SparseLu::new(Arc::new(ordering)), entries)
}

/// A sparse matrix whose rows are strictly diagonally dominant, then shuffled, so that its
/// large entries lie off the diagonal and a matching must find them.
fn shuffled_dominant() -> impl Strategy<Value = Matrix<f64>> {
    (1usize..40).prop_flat_map(|dimension| {
        let off =
            proptest::collection::vec((0..dimension, 0..dimension, -1.0..1.0f64), 0..4 * dimension);
        let signs = proptest::collection::vec(any::<bool>(), dimension);
        let order = Just((0..dimension).collect::<Vec<_>>()).prop_shuffle();
        (off, signs, order).prop_map(move |(off, signs, order)| {
            let mut sums = vec![1.0; dimension];
            let mut positions = Vec::new();
            let mut values = Vec::new();
            for (row, column, value) in off.into_iter().filter(|(row, column, _)| row != column) {
                sums[row] += f64::abs(value);
                positions.push((order[row], column));
                values.push(value);
            }
            for (row, sum) in sums.into_iter().enumerate() {
                positions.push((order[row], row));
                values.push(if signs[row] { sum } else { -sum });
            }
            Matrix {
                dimension,
                positions,
                values,
            }
        })
    })
}

/// A sparse matrix with no structure to lean on: entries of random values, from −1 to 1, at
/// random positions, and at the positions of a random permutation, so that some matching
/// covers every column, though not necessarily with large entries.
fn random_sparse() -> impl Strategy<Value = Matrix<f64>> {
    (1usize..40).prop_flat_map(|dimension| {
        let entries =
            proptest::collection::vec((0..dimension, 0..dimension, -1.0..1.0f64), 0..4 * dimension);
        let matched = proptest::collection::vec(-1.0..1.0f64, dimension);
        let order = Just((0..dimension).collect::<Vec<_>>()).prop_shuffle();
        (entries, matched, order).prop_map(move |(entries, matched, order)| {
            let (mut positions, mut values): (Vec<_>, Vec<_>) = entries
                .into_iter()
                .map(|(row, column, value)| ((row, column), value))
                .unzip();
            positions.extend(order.iter().enumerate().map(|(column, &row)| (row, column)));
            values.extend(matched);
            Matrix {
                dimension,
                positions,
                values,
            }
        })
    })
}

/// Two matrices of one pattern, each strictly diagonally dominant by rows once its rows are
/// put back in order, the first shuffled by one permutation and the second by another, so
/// that the pivots the first takes are mostly too small for the second, from a column
/// anywhere in the order of elimination.
fn reshuffled_pair() -> impl Strategy<Value = (Matrix<f64>, Matrix<f64>)> {
    (2usize..30).prop_flat_map(|dimension| {
        let off =
            proptest::collection::vec((0..dimension, 0..dimension, -1.0..1.0f64), 0..3 * dimension);
        let order = || Just((0..dimension).collect::<Vec<_>>()).prop_shuffle();
        (off, order(), order()).prop_map(move |(off, first, second)| {
            let off: Vec<_> = off
                .into_iter()
                .filter(|(row, column, _)| row != column)
                .collect();
            let mut sums = vec![1.0; dimension];
            for &(row, _, value) in &off {
                sums[row] += f64::abs(value);
            }
            // Each matrix's own entries, then the other's positions with zeros, so that both
            // claim the same positions in the same order. The second's dominant entries are
            // twice as large, so that it differs from the first even where the two
            // permutations agree.
            let shuffled = |order: &[usize], scale: f64| -> Vec<((usize, usize), f64)> {
                off.iter()
                    .map(|&(row, column, value)| ((order[row], column), value))
                    .chain(
                        sums.iter()
                            .enumerate()
                            .map(|(row, &sum)| ((order[row], row), scale * sum)),
                    )
                    .collect()
            };
            let (one, other) = (shuffled(&first, 1.0), shuffled(&second, 2.0));
            let matrix =
                |own: &[((usize, usize), f64)], zeros: &[((usize, usize), f64)], first: bool| {
                    let zeros = zeros.iter().map(|&(position, _)| (position, 0.0));
                    let entries: Vec<_> = if first {
                        own.iter().copied().chain(zeros).collect()
                    } else {
                        zeros.chain(own.iter().copied()).collect()
                    };
                    let (positions, values) = entries.into_iter().unzip();
                    Matrix {
                        dimension,
                        positions,
                        values,
                    }
                };
            (matrix(&one, &other, true), matrix(&other, &one, false))
        })
    })
}

/// The modified nodal equations of a random circuit, `G + s C`, on `nodes` nodes and the
/// reference, numbered `nodes`, with a current unknown per voltage source after the nodes.
#[derive(Debug, Clone)]
struct Circuit {
    nodes: usize,
    /// Conductances, in siemens, between two nodes.
    conductances: Vec<(usize, usize, f64)>,
    /// Capacitances, in farads, between two nodes.
    capacitances: Vec<(usize, usize, f64)>,
    /// Voltage sources between a positive and a negative node, their loops none.
    sources: Vec<(usize, usize)>,
    /// Transconductances, in siemens: from a drain to a source, a current of the
    /// transconductance times the voltage from a gate to the source.
    transconductances: Vec<(usize, usize, usize, f64)>,
    /// A factor for each element's value in a second matrix of the same pattern, taken in
    /// turn.
    changes: Vec<f64>,
}

impl Circuit {
    fn dimension(&self) -> usize {
        self.nodes + self.sources.len()
    }

    /// The matrix `G + s C`, its element values scaled by `changes` if `changed`.
    fn matrix<T: Scalar + One + From<f64>>(&self, s: T, changed: bool) -> Matrix<T> {
        let reference = self.nodes;
        let mut matrix = Matrix {
            dimension: self.dimension(),
            positions: Vec::new(),
            values: Vec::new(),
        };
        let mut element = 0;
        let mut scale = |value: f64| {
            element += 1;
            if changed {
                value * self.changes[element % self.changes.len()]
            } else {
                value
            }
        };
        // The unknown of a node: none for the reference.
        let node = |node: usize| (node != reference).then_some(node);
        let mut stamp = |row: Option<usize>, column: Option<usize>, value: T| {
            if let (Some(row), Some(column)) = (row, column) {
                matrix.positions.push((row, column));
                matrix.values.push(value);
            }
        };
        // Every node's diagonal, as an engine claims it for its shunts.
        for index in 0..self.nodes {
            stamp(Some(index), Some(index), T::ZERO);
        }
        let mut admittances = Vec::new();
        for &(a, b, g) in &self.conductances {
            admittances.push((a, b, T::from(scale(g))));
        }
        for &(a, b, c) in &self.capacitances {
            admittances.push((a, b, s * T::from(scale(c))));
        }
        for (a, b, value) in admittances {
            let (a, b) = (node(a), node(b));
            stamp(a, a, value);
            stamp(b, b, value);
            stamp(a, b, -value);
            stamp(b, a, -value);
        }
        for &(drain, source, gate, gm) in &self.transconductances {
            let gm = T::from(scale(gm));
            let (drain, source, gate) = (node(drain), node(source), node(gate));
            stamp(drain, gate, gm);
            stamp(drain, source, -gm);
            stamp(source, gate, -gm);
            stamp(source, source, gm);
        }
        for (index, &(positive, negative)) in self.sources.iter().enumerate() {
            let branch = Some(self.nodes + index);
            let (positive, negative) = (node(positive), node(negative));
            stamp(positive, branch, T::ONE);
            stamp(negative, branch, -T::ONE);
            stamp(branch, positive, T::ONE);
            stamp(branch, negative, -T::ONE);
        }
        matrix
    }
}

/// A random circuit: a tree of resistors joins every node to the reference, more resistors
/// and capacitors join random nodes, voltage sources form a forest, each from a node to an
/// earlier one or the reference, transconductances couple random nodes, and, half the time,
/// a supply node held by a source feeds many nodes through resistors.
fn circuit() -> impl Strategy<Value = Circuit> {
    (1usize..30, any::<bool>()).prop_flat_map(|(nodes, supplied)| {
        let total = nodes + usize::from(supplied);
        let tree = proptest::collection::vec((any::<Index>(), -3.0..0.0f64), nodes);
        let extra = proptest::collection::vec((0..=total, 0..=total, -3.0..0.0f64), 0..2 * total);
        let capacitors =
            proptest::collection::vec((0..=total, 0..=total, -3.0..0.0f64), 0..2 * total);
        let sources = proptest::collection::btree_map(0..nodes, any::<Index>(), 0..=nodes / 2);
        let gains =
            proptest::collection::vec((0..=total, 0..=total, 0..=total, -3.0..-1.0f64), 0..nodes);
        let fed = proptest::collection::btree_set(0..nodes, 0..=nodes);
        let changes = proptest::collection::vec(-2.0..2.0f64, 1..16);
        (tree, extra, capacitors, sources, gains, fed, changes).prop_map(
            move |(tree, extra, capacitors, sources, gains, fed, changes)| {
                let reference = total;
                let power = |exponent: f64| 10f64.powf(exponent);
                let mut conductances: Vec<(usize, usize, f64)> = tree
                    .into_iter()
                    .enumerate()
                    .map(|(node, (parent, exponent))| {
                        let parent = parent.index(node + 1);
                        (
                            node,
                            if parent == node { reference } else { parent },
                            power(exponent),
                        )
                    })
                    .collect();
                conductances.extend(
                    extra
                        .into_iter()
                        .filter(|(a, b, _)| a != b)
                        .map(|(a, b, exponent)| (a, b, power(exponent))),
                );
                let mut sources: Vec<(usize, usize)> = sources
                    .into_iter()
                    .map(|(positive, negative)| {
                        let negative = negative.index(positive + 1);
                        (
                            positive,
                            if negative == positive {
                                reference
                            } else {
                                negative
                            },
                        )
                    })
                    .collect();
                if supplied {
                    let supply = nodes;
                    sources.push((supply, reference));
                    conductances.extend(fed.into_iter().map(|node| (supply, node, 0.1)));
                }
                Circuit {
                    nodes: total,
                    conductances,
                    capacitances: capacitors
                        .into_iter()
                        .filter(|(a, b, _)| a != b)
                        .map(|(a, b, exponent)| (a, b, power(exponent)))
                        .collect(),
                    sources,
                    transconductances: gains
                        .into_iter()
                        .map(|(drain, source, gate, exponent)| {
                            (drain, source, gate, power(exponent))
                        })
                        .collect(),
                    changes: changes.into_iter().map(power).collect(),
                }
            },
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Shuffled rows put the large entries off the diagonal: threshold pivoting must find
    /// them, the matching's diagonal holding small entries or none. The small entries it
    /// takes while they hold a tenth of their column let the factors' entries grow, so that
    /// the backward errors of these solutions, below 10⁻¹⁴ for all but one in a thousand,
    /// have reached 1.4·10⁻¹² in 40,000: they are held to 10⁻¹⁰.
    #[test]
    fn solves_matrices_whose_large_entries_are_off_the_diagonal(matrix in shuffled_dominant()) {
        let (mut lu, _) = solver(&matrix);
        check_within(&mut lu, &matrix, 1e-10)?;
    }

    /// Matrices without dominance or symmetry, whose pivots the threshold search must find
    /// wherever they are.
    #[test]
    fn solves_random_sparse_matrices(matrix in random_sparse()) {
        let (mut lu, _) = solver(&matrix);
        check(&mut lu, &matrix)?;
    }

    /// A refactorization whose kept pivots become too small searches for the pivots afresh
    /// from the first of them, keeping the columns before it, and solves as well as a first
    /// factorization.
    #[test]
    fn searches_afresh_from_a_kept_pivot_too_small((first, second) in reshuffled_pair()) {
        let (mut lu, _) = solver(&first);
        check(&mut lu, &first)?;
        check(&mut lu, &second)?;
        let statistics = lu.statistics();
        prop_assert_eq!(statistics.factorizations + statistics.refactorizations, 2);
    }

    /// The equations of circuits: voltage sources' equations with nothing on their
    /// diagonals, supplies coupled to many nodes, and transconductances out of symmetry.
    #[test]
    fn solves_the_equations_of_circuits(circuit in circuit(), alpha in -1.0..3.0f64) {
        let matrix = circuit.matrix(10f64.powf(alpha), false);
        let (mut lu, _) = solver(&matrix);
        check(&mut lu, &matrix)?;
        prop_assert_eq!(lu.statistics().factorizations, 1);
    }

    /// Complex equations of circuits at a frequency, `G + jωC`, solved in complex numbers.
    #[test]
    fn solves_complex_equations(circuit in circuit(), omega in -1.0..3.0f64) {
        let matrix = circuit.matrix(Complex64::new(0.0, 10f64.powf(omega)), false);
        let (mut lu, _) = solver(&matrix);
        check(&mut lu, &matrix)?;
    }

    /// New values of the same pattern refactor with the pivots of the first, and solve as
    /// well as a factorization afresh; the same values again keep the factors.
    #[test]
    fn refactors_new_values_of_a_pattern(circuit in circuit(), alpha in -1.0..3.0f64) {
        let first = circuit.matrix(10f64.powf(alpha), false);
        let second = circuit.matrix(10f64.powf(alpha), true);
        let (mut lu, _) = solver(&first);
        check(&mut lu, &first)?;
        check(&mut lu, &second)?;
        check(&mut lu, &second)?;
        let statistics = lu.statistics();
        prop_assert_eq!(statistics.factorizations + statistics.refactorizations, 2);
        prop_assert_eq!(statistics.reuses, 1);
    }

    /// Complex equations at another frequency, with other element values, refactor with the
    /// pivots of the first, as a small-signal analysis does from one frequency to the next.
    #[test]
    fn refactors_complex_equations_at_another_frequency(
        circuit in circuit(),
        omegas in (-1.0..3.0f64, -1.0..3.0f64),
    ) {
        let first = circuit.matrix(Complex64::new(0.0, 10f64.powf(omegas.0)), false);
        let second = circuit.matrix(Complex64::new(0.0, 10f64.powf(omegas.1)), true);
        let (mut lu, _) = solver(&first);
        check(&mut lu, &first)?;
        check(&mut lu, &second)?;
        let statistics = lu.statistics();
        prop_assert_eq!(statistics.factorizations + statistics.refactorizations, 2);
    }

    /// The real and the complex solvers of one pattern share its ordering.
    #[test]
    fn real_and_complex_solvers_share_an_ordering(circuit in circuit()) {
        let real = circuit.matrix(1.0, false);
        let complex = circuit.matrix(Complex64::new(0.0, 1.0), false);
        let (pattern, _) = real.pattern();
        let ordering = Arc::new(Ordering::new(pattern).expect("fits"));
        let mut real_lu = SparseLu::new(Arc::clone(&ordering));
        let mut complex_lu = SparseLu::new(Arc::clone(&ordering));
        check(&mut real_lu, &real)?;
        check(&mut complex_lu, &complex)?;
        prop_assert!(Arc::ptr_eq(real_lu.ordering(), complex_lu.ordering()));
    }
}

/// A dense 2 × 2 matrix's pattern and solver, with its values by slot.
fn two_by_two(rows: [[f64; 2]; 2]) -> (SparseLu<f64>, Vec<f64>, Matrix<f64>) {
    let matrix = Matrix {
        dimension: 2,
        positions: vec![(0, 0), (0, 1), (1, 0), (1, 1)],
        values: rows.into_iter().flatten().collect(),
    };
    let (lu, entries) = solver(&matrix);
    let stored = matrix.stored(lu.pattern(), &entries);
    (lu, stored, matrix)
}

/// The solution of `matrix x = [1, 2]` by `lu`.
fn solution(lu: &mut SparseLu<f64>, matrix: &Matrix<f64>) -> Result<Vec<f64>, LinearError> {
    let (_, entries) = matrix.pattern();
    let stored = matrix.stored(lu.pattern(), &entries);
    let mut x = vec![1.0, 2.0];
    lu.factor(&stored)?.solve(&mut x)?;
    Ok(x)
}

#[test]
fn falls_back_to_a_search_when_a_kept_pivot_becomes_too_small() {
    // The first matrix pivots on its diagonal; in the second, either diagonal entry is
    // 10⁻¹⁴ of the entry beside it in its column, which the refactorization refuses for the
    // first column it eliminates, searching afresh from there.
    let (mut lu, _, first) = two_by_two([[4.0, 1.0], [1.0, 3.0]]);
    let x = solution(&mut lu, &first).unwrap();
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-15 && (x[1] - 7.0 / 11.0).abs() < 1e-15);
    // By Cramer's rule, [[a, b], [c, d]] x = [1, 2] has x = [d − 2b, 2a − c] / (ad − bc).
    let small = 1e-14;
    let second = Matrix {
        values: vec![small, 1.0, 1.0, small],
        ..first.clone()
    };
    let x = solution(&mut lu, &second).unwrap();
    let determinant = small * small - 1.0;
    let wanted = [
        (small - 2.0) / determinant,
        (2.0 * small - 1.0) / determinant,
    ];
    assert!(
        (x[0] - wanted[0]).abs() < 1e-15 && (x[1] - wanted[1]).abs() < 1e-15,
        "{x:?} against {wanted:?}"
    );
    assert_eq!(
        lu.statistics(),
        Statistics {
            factorizations: 2,
            refactorizations: 0,
            reuses: 0
        }
    );
    // With values the new pivots suit, the refactorization keeps them.
    let third = Matrix {
        values: vec![1e-4, 2.0, 1.0, 1e-4],
        ..first
    };
    let x = solution(&mut lu, &third).unwrap();
    let determinant = 1e-8 - 2.0;
    let wanted = [(1e-4 - 4.0) / determinant, (2e-4 - 1.0) / determinant];
    assert!(
        (x[0] - wanted[0]).abs() < 1e-15 && (x[1] - wanted[1]).abs() < 1e-15,
        "{x:?} against {wanted:?}"
    );
    assert_eq!(lu.statistics().refactorizations, 1);
}

#[test]
fn pivots_off_the_diagonal_where_it_holds_too_little() {
    let (mut lu, stored, _) = two_by_two([[1e-12, 1.0], [2.0, 1e-12]]);
    let mut x = [1.0, 8.0];
    lu.factor(&stored).unwrap().solve(&mut x).unwrap();
    let determinant = 1e-24 - 2.0;
    let wanted = [(1e-12 - 8.0) / determinant, (8e-12 - 2.0) / determinant];
    assert!(
        (x[0] - wanted[0]).abs() < 1e-15 && (x[1] - wanted[1]).abs() < 1e-15,
        "{x:?} against {wanted:?}"
    );
}

/// Whether `result` is a singular error.
fn is_singular<T>(result: &Result<T, LinearError>) -> bool {
    matches!(result, Err(LinearError::Singular { .. }))
}

#[test]
fn reports_singular_matrices() {
    // Numerically singular: the second row is twice the first.
    let (mut lu, stored, _) = two_by_two([[1.0, 2.0], [2.0, 4.0]]);
    assert!(is_singular(&lu.factor(&stored)));
    // A column of zeros is the one named, whatever the order of elimination.
    let (mut lu, stored, _) = two_by_two([[1.0, 0.0], [3.0, 0.0]]);
    assert_eq!(
        lu.factor(&stored).err(),
        Some(LinearError::Singular { columns: vec![1] })
    );
    // A refactorization that meets a singular matrix fails too, and the solver recovers.
    let (mut lu, stored, _) = two_by_two([[1.0, 2.0], [3.0, 4.0]]);
    lu.factor(&stored).unwrap();
    assert!(is_singular(&lu.factor(&[1.0, 2.0, 2.0, 4.0])));
    let mut x = [1.0, 1.0];
    lu.factor(&stored).unwrap().solve(&mut x).unwrap();
    assert!(
        (x[0] + 1.0).abs() < 1e-15 && (x[1] - 1.0).abs() < 1e-15,
        "{x:?}"
    );
    // A column without a stored position is singular for any values.
    let mut builder = PatternBuilder::new();
    builder.entry(0, 0);
    builder.entry(1, 0);
    let ordering = Ordering::new(Arc::new(builder.finish(2))).unwrap();
    assert!(ordering.is_singular());
    let mut lu = SparseLu::new(Arc::new(ordering));
    assert_eq!(
        lu.factor(&[1.0, 1.0]).err(),
        Some(LinearError::Singular { columns: vec![1] })
    );
    // Two voltage sources in parallel: their equations read only the same node, so their
    // currents, unknowns 1 and 2, appear in the node's equation alone.
    let mut builder = PatternBuilder::new();
    for (row, column) in [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)] {
        builder.entry(row, column);
    }
    let ordering = Ordering::new(Arc::new(builder.finish(3))).unwrap();
    assert_eq!(ordering.undetermined(), [1, 2]);
    let mut lu = SparseLu::new(Arc::new(ordering));
    assert_eq!(
        lu.factor(&[1.0; 5]).err(),
        Some(LinearError::Singular {
            columns: vec![1, 2]
        })
    );
}

#[test]
fn reports_solutions_that_are_not_finite() {
    // A subnormal pivot is no zero, but its inverse overflows.
    let diagonal = Matrix {
        dimension: 2,
        positions: vec![(0, 0), (1, 1)],
        values: vec![1.0, 1e-310],
    };
    let (mut lu, entries) = solver(&diagonal);
    let stored = diagonal.stored(lu.pattern(), &entries);
    let mut x = [1.0, 1.0];
    assert_eq!(
        lu.factor(&stored).unwrap().solve(&mut x),
        Err(LinearError::Singular { columns: vec![1] })
    );
}

#[test]
fn solves_nearly_singular_matrices() {
    // Rows apart by 10⁻¹⁰: the condition number is about 4·10¹⁰.
    let gap = 1e-10;
    let (mut lu, stored, matrix) = two_by_two([[1.0, 1.0], [1.0, 1.0 + gap]]);
    let mut x = [2.0, 2.0 + gap];
    lu.factor(&stored).unwrap().solve(&mut x).unwrap();
    let dense = matrix.dense();
    assert!(backward_error(&dense, &x, &[2.0, 2.0 + gap]) < 1e-15);
    assert!(
        (x[0] - 1.0).abs() < 1e-5 && (x[1] - 1.0).abs() < 1e-5,
        "{x:?}"
    );
}

#[test]
fn keeps_the_factors_of_the_very_matrix_last_factored() {
    let (mut lu, stored, _) = two_by_two([[2.0, 1.0], [1.0, 2.0]]);
    lu.factor(&stored).unwrap();
    lu.factor(&stored).unwrap();
    let mut changed = stored.clone();
    changed[0] = 3.0;
    lu.factor(&changed).unwrap();
    assert_eq!(
        lu.statistics(),
        Statistics {
            factorizations: 1,
            refactorizations: 1,
            reuses: 1
        }
    );
}

#[test]
fn solves_the_empty_system() {
    let ordering = Ordering::new(Arc::new(PatternBuilder::new().finish(0))).unwrap();
    assert_eq!(ordering.blocks(), 0);
    let mut lu = SparseLu::<f64>::new(Arc::new(ordering));
    lu.factor(&[]).unwrap().solve(&mut []).unwrap();
}

/// A ladder of `stages` resistors from a supply node to the reference, each of its nodes
/// fed from the supply through a resistor of its own as well, with a capacitor to the
/// reference, the supply held by a voltage source: the supply's node joins every other.
fn supplied_ladder(stages: usize) -> Matrix<f64> {
    let supply = stages;
    let mut conductances = Vec::new();
    for node in 0..stages {
        conductances.push((node, if node == 0 { stages + 1 } else { node - 1 }, 1.0));
        conductances.push((node, supply, 0.5));
        conductances.push((node, stages + 1, 0.25));
    }
    let circuit = Circuit {
        nodes: stages + 1,
        conductances,
        capacitances: Vec::new(),
        sources: vec![(supply, stages + 1)],
        transconductances: Vec::new(),
        changes: vec![1.0],
    };
    circuit.matrix(1.0, false)
}

#[test]
fn a_supply_joining_every_node_fills_nothing() {
    // The supply's node and its source's current become blocks of their own, so the
    // supply's entries stay above the ladder's block, which, tridiagonal, fills nothing.
    let matrix = supplied_ladder(200);
    let (mut lu, entries) = solver(&matrix);
    assert_eq!(lu.ordering().blocks(), 3);
    let stored = matrix.stored(lu.pattern(), &entries);
    let mut x = rhs::<f64>(matrix.dimension);
    let b = x.clone();
    lu.factor(&stored).unwrap().solve(&mut x).unwrap();
    assert!(backward_error(&matrix.dense(), &x, &b) < 1e-15);
    assert_eq!(lu.entries(), lu.pattern().len());
}
