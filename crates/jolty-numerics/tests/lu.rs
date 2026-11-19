//! Integration tests comparing the sparse LU solver against a dense Gauss-Jordan reference.
//!
//! Coverage spans random off-diagonal matrices, real and complex circuit matrices,
//! refactorization after value updates, Meyer displacement and series-resistance condensation,
//! nested dissection, and singular or ill-conditioned systems.
#![cfg(test)]

use std::sync::Arc;

use jolty_numerics::{
    Complex64, Condensation, Dissecting, Entry, LinearError, Ordering, Pattern, PatternBuilder,
    Scalar, SparseLu, Work, Workspace,
};
use proptest::prelude::*;
use proptest::sample::Index;

/// Coordinate-list representation of a square matrix where repeated positions sum.
#[derive(Debug, Clone)]
struct Matrix<T> {
    dimension: usize,
    positions: Vec<(usize, usize)>,
    values: Vec<T>,
}

impl<T: Scalar> Matrix<T> {
    /// Builds a [`Pattern`] from the recorded positions and returns the corresponding entries.
    fn pattern(&self) -> (Arc<Pattern>, Vec<Entry>) {
        let mut builder = PatternBuilder::new();
        let entries = self
            .positions
            .iter()
            .map(|&(row, column)| builder.entry(row, column))
            .collect();
        (Arc::new(builder.finish(self.dimension)), entries)
    }

    /// Scatters matrix values into a slot-indexed slice matching `pattern` and `entries`.
    fn stored(&self, pattern: &Pattern, entries: &[Entry]) -> Vec<T> {
        let mut stored = vec![T::ZERO; pattern.len()];
        for (&entry, &value) in entries.iter().zip(&self.values) {
            stored[pattern.slot(entry)] += value;
        }
        stored
    }

    /// Materializes the matrix as a dense row-major `Vec<Vec<T>>`.
    fn dense(&self) -> Vec<Vec<T>> {
        let mut dense = vec![vec![T::ZERO; self.dimension]; self.dimension];
        for (&(row, column), &value) in self.positions.iter().zip(&self.values) {
            dense[row][column] += value;
        }
        dense
    }
}

/// Returns the transpose of dense `matrix`.
fn transposed<T: Scalar>(matrix: &[Vec<T>]) -> Vec<Vec<T>> {
    (0..matrix.len())
        .map(|row| matrix.iter().map(|other| other[row]).collect())
        .collect()
}

/// Computes the dense matrix-vector product `matrix * x`.
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

/// Returns the maximum element magnitude in `values`.
fn largest<T: Scalar>(values: &[T]) -> f64 {
    values
        .iter()
        .map(|value| value.magnitude())
        .fold(0.0, f64::max)
}

/// Returns the infinity norm (maximum row sum of magnitudes) of dense `matrix`.
fn norm<T: Scalar>(matrix: &[Vec<T>]) -> f64 {
    matrix
        .iter()
        .map(|row| row.iter().map(|value| value.magnitude()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Computes the inverse of dense `matrix` via Gauss-Jordan elimination with partial pivoting,
/// returning `None` if singular.
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

/// Multiplicative identity trait for test scalar types.
trait One {
    const ONE: Self;
}

impl One for f64 {
    const ONE: Self = 1.0;
}

impl One for Complex64 {
    const ONE: Self = Complex64::new(1.0, 0.0);
}

/// Builds a deterministic nonzero right-hand side vector of length `dimension`.
fn right_hand_side<T: Scalar + One>(dimension: usize) -> Vec<T> {
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

/// Computes the relative backward error of `x` solving `matrix * x = b`.
fn backward_error<T: Scalar>(matrix: &[Vec<T>], x: &[T], b: &[T]) -> f64 {
    let residual: Vec<T> = product(matrix, x)
        .into_iter()
        .zip(b)
        .map(|(found, &wanted)| found - wanted)
        .collect();
    largest(&residual) / (norm(matrix) * largest(x) + largest(b))
}

/// Verifies that `lu` solves `matrix` and its transpose against the dense reference within a
/// backward error of `1e-12`.
fn check<T: Scalar + One>(lu: &mut SparseLu<T>, matrix: &Matrix<T>) -> Result<(), TestCaseError> {
    check_within(lu, matrix, 1e-12)
}

/// Verifies that `lu` solves `matrix` and its transpose within `backward` and within
/// `backward * condition` relative forward error.
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
    let b: Vec<T> = right_hand_side(matrix.dimension);
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

/// Builds a [`SparseLu`] solver and entry handles for the pattern of `matrix`.
fn solver<T: Scalar>(matrix: &Matrix<T>) -> (SparseLu<T>, Vec<Entry>) {
    let (pattern, entries) = matrix.pattern();
    let ordering = Ordering::new(pattern).expect("fits");
    (SparseLu::new(Arc::new(ordering)), entries)
}

/// Generates sparse row-diagonally-dominant matrices with shuffled rows so large pivots sit off
/// the diagonal.
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

/// Generates unstructured sparse matrices with random entries in `[-1, 1]` plus a random
/// permutation ensuring a structural matching.
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

/// Generates two matrices sharing one sparsity pattern but using distinct row shuffles of a
/// diagonally dominant matrix so pivots from the first frequently fail threshold checks in the
/// second.
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

/// Random modified-nodal circuit generating `G + s C` over `nodes` non-ground nodes,
/// ground node `nodes`, and trailing branch unknowns.
#[derive(Debug, Clone)]
struct Circuit {
    nodes: usize,
    /// Conductances, in siemens, between two nodes.
    conductances: Vec<(usize, usize, f64)>,
    /// Capacitances, in farads, between two nodes.
    capacitances: Vec<(usize, usize, f64)>,
    /// Acyclic voltage sources between a positive and a negative node.
    sources: Vec<(usize, usize)>,
    /// Transconductances, in siemens: from a drain to a source, injecting `gm * (v_gate - v_source)`.
    transconductances: Vec<(usize, usize, usize, f64)>,
    /// Meyer displacement currents appended after voltage-source currents:
    /// `(gate, node, oxide_capacitance, fraction)` enforcing `w = Cox * d(v_gate - v_node)/dt`
    /// and injecting `f * w`.
    displacements: Vec<(usize, usize, f64, f64)>,
    /// Small series resistances from a terminal to a private internal node coupled via
    /// conductance and Meyer displacement: `(terminal, other, conductance, resistance, oxide,
    /// fraction)` with resistance below 1 ohm.
    series: Vec<(usize, usize, f64, f64, f64, f64)>,
    /// Per-element scaling factors applied cyclically when generating a perturbed matrix.
    changes: Vec<f64>,
}

impl Circuit {
    fn dimension(&self) -> usize {
        self.nodes + self.sources.len() + self.displacements.len() + 3 * self.series.len()
    }

    /// Returns the unknown indices of standalone Meyer displacements.
    fn displaced(&self) -> Vec<usize> {
        let first = self.nodes + self.sources.len();
        (first..first + self.displacements.len()).collect()
    }

    /// Returns the first unknown index of the series-resistance triples (internal node, current,
    /// displacement).
    fn first_series(&self) -> usize {
        self.nodes + self.sources.len() + self.displacements.len()
    }

    /// Returns guaranteed-dominant `(row, column)` condensation pivots for each standalone
    /// displacement and series-resistance pair.
    fn pivots(&self) -> Vec<(usize, usize)> {
        let mut pivots: Vec<(usize, usize)> = self
            .displaced()
            .into_iter()
            .map(|unknown| (unknown, unknown))
            .collect();
        for index in 0..self.series.len() {
            let inner = self.first_series() + 3 * index;
            pivots.extend([(inner, inner + 1), (inner + 2, inner + 2)]);
        }
        pivots
    }

    /// Stamps `G + s C`, scaling element values by `changes` when `changed` is `true`.
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
        for (unknown, &(gate, other, oxide, fraction)) in
            self.displaced().into_iter().zip(&self.displacements)
        {
            let (own, gate, other) = (Some(unknown), node(gate), node(other));
            let oxide = s * T::from(scale(oxide));
            stamp(own, own, T::ONE);
            stamp(own, gate, -oxide);
            stamp(own, other, oxide);
            stamp(gate, own, T::from(fraction));
            stamp(other, own, -T::from(fraction));
        }
        for (index, &(terminal, other, conductance, resistance, oxide, fraction)) in
            self.series.iter().enumerate()
        {
            let inner = self.first_series() + 3 * index;
            let (current, own) = (Some(inner + 1), Some(inner + 2));
            let (terminal, other, inner) = (node(terminal), node(other), Some(inner));
            let conductance = T::from(scale(conductance));
            stamp(inner, inner, conductance);
            stamp(other, other, conductance);
            stamp(inner, other, -conductance);
            stamp(other, inner, -conductance);
            stamp(terminal, current, T::ONE);
            stamp(inner, current, -T::ONE);
            stamp(current, terminal, T::ONE);
            stamp(current, inner, -T::ONE);
            stamp(current, current, -T::from(scale(resistance)));
            let oxide = s * T::from(scale(oxide));
            stamp(own, own, T::ONE);
            stamp(own, other, -oxide);
            stamp(own, inner, oxide);
            stamp(other, own, T::from(fraction));
            stamp(inner, own, -T::from(fraction));
        }
        matrix
    }
}

/// Generates a connected random circuit with a spanning resistor tree, extra passive elements,
/// a voltage-source forest, transconductances, and an optional supply rail.
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
                    displacements: Vec::new(),
                    series: Vec::new(),
                    changes: changes.into_iter().map(power).collect(),
                }
            },
        )
    })
}

/// Generates a random circuit extended with Meyer oxide displacement currents between random
/// gates and nodes.
fn meyer() -> impl Strategy<Value = Circuit> {
    circuit().prop_flat_map(|circuit| {
        let total = circuit.nodes;
        let displacements = proptest::collection::vec(
            (0..=total, 0..=total, -3.0..0.0f64, 0.0..=1.0f64),
            1..2 * total + 2,
        );
        (Just(circuit), displacements).prop_map(|(mut circuit, displacements)| {
            circuit.displacements = displacements
                .into_iter()
                .map(|(gate, other, exponent, fraction)| {
                    (gate, other, 10f64.powf(exponent), fraction)
                })
                .collect();
            circuit
        })
    })
}

/// Generates a random circuit with Meyer displacements and small series resistances (1 milliohm
/// to 1 ohm) coupled to internal switched channels.
fn switched() -> impl Strategy<Value = Circuit> {
    meyer().prop_flat_map(|circuit| {
        let total = circuit.nodes;
        let series = proptest::collection::vec(
            (
                0..=total,
                0..=total,
                (-1.0..2.0f64, -3.0..0.0f64),
                (-3.0..0.0f64, 0.0..=1.0f64),
            ),
            1..total + 2,
        );
        (Just(circuit), series).prop_map(|(mut circuit, series)| {
            circuit.series = series
                .into_iter()
                .map(
                    |(terminal, other, (conductance, resistance), (oxide, fraction))| {
                        (
                            terminal,
                            other,
                            10f64.powf(conductance),
                            10f64.powf(resistance),
                            10f64.powf(oxide),
                            fraction,
                        )
                    },
                )
                .collect();
            circuit
        })
    })
}

/// Builds a [`SparseLu`] solver condensing the displacements and series currents of `circuit`,
/// optionally dissecting blocks to `interiors`.
fn condensing<T: Scalar>(
    circuit: &Circuit,
    matrix: &Matrix<T>,
    interiors: Option<usize>,
) -> SparseLu<T> {
    let (pattern, _) = matrix.pattern();
    let condensation = Arc::new(Condensation::new(pattern, &circuit.pivots()));
    let ordering =
        Ordering::condensed(condensation, interiors.map(Dissecting::Every)).expect("fits");
    SparseLu::new(Arc::new(ordering))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Verifies threshold pivoting on row-shuffled dominant matrices where large entries sit
    /// off the diagonal.
    ///
    /// Accepting pivots down to a tenth of the column maximum allows mild factor growth, so
    /// the backward error bound is relaxed to `1e-10`.
    #[test]
    fn solves_matrices_whose_large_entries_are_off_the_diagonal(matrix in shuffled_dominant()) {
        let (mut lu, _) = solver(&matrix);
        check_within(&mut lu, &matrix, 1e-10)?;
    }

    /// Verifies threshold pivoting on unstructured random sparse matrices without dominance or
    /// symmetry.
    #[test]
    fn solves_random_sparse_matrices(matrix in random_sparse()) {
        let (mut lu, _) = solver(&matrix);
        check(&mut lu, &matrix)?;
    }

    /// Verifies that refactorization falls back to searching afresh from the first rejected
    /// pivot while matching a fresh factorization.
    #[test]
    fn searches_afresh_from_a_kept_pivot_too_small((first, second) in reshuffled_pair()) {
        let (mut lu, _) = solver(&first);
        check(&mut lu, &first)?;
        check(&mut lu, &second)?;
        let statistics = lu.statistics();
        prop_assert_eq!(statistics.factorizations, 1);
        prop_assert_eq!(statistics.refactorizations + statistics.fallbacks, 1);
    }

    /// Verifies real circuit equations containing zero-diagonal voltage sources, supply nodes,
    /// and asymmetric transconductances.
    #[test]
    fn solves_the_equations_of_circuits(circuit in circuit(), alpha in -1.0..3.0f64) {
        let matrix = circuit.matrix(10f64.powf(alpha), false);
        let (mut lu, _) = solver(&matrix);
        check(&mut lu, &matrix)?;
        prop_assert_eq!(lu.statistics().factorizations, 1);
    }

    /// Verifies complex circuit equations `G + jωC` at arbitrary frequencies.
    #[test]
    fn solves_complex_equations(circuit in circuit(), omega in -1.0..3.0f64) {
        let matrix = circuit.matrix(Complex64::new(0.0, 10f64.powf(omega)), false);
        let (mut lu, _) = solver(&matrix);
        check(&mut lu, &matrix)?;
    }

    /// Verifies that updated values refactor using cached pivots and repeated identical values
    /// reuse existing factors.
    #[test]
    fn refactors_new_values_of_a_pattern(circuit in circuit(), alpha in -1.0..3.0f64) {
        let first = circuit.matrix(10f64.powf(alpha), false);
        let second = circuit.matrix(10f64.powf(alpha), true);
        let (mut lu, _) = solver(&first);
        check(&mut lu, &first)?;
        check(&mut lu, &second)?;
        check(&mut lu, &second)?;
        let statistics = lu.statistics();
        prop_assert_eq!(statistics.factorizations, 1);
        prop_assert_eq!(statistics.refactorizations + statistics.fallbacks, 1);
        prop_assert_eq!(statistics.reuses, 1);
    }

    /// Verifies that complex circuit matrices refactor across frequency and parameter changes.
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
        prop_assert_eq!(statistics.factorizations, 1);
        prop_assert_eq!(statistics.refactorizations + statistics.fallbacks, 1);
    }

    /// Verifies that a reset solver forgets cached pivots and produces bit-identical results to
    /// a freshly constructed solver.
    #[test]
    fn a_reset_solver_factors_as_a_new_one((first, second) in reshuffled_pair()) {
        let solved = |lu: &mut SparseLu<f64>, matrix: &Matrix<f64>| {
            let (_, entries) = matrix.pattern();
            let stored = matrix.stored(lu.pattern(), &entries);
            let mut x: Vec<f64> = right_hand_side(matrix.dimension);
            lu.factor(&stored)?.solve(&mut x)?;
            Ok::<Vec<u64>, LinearError>(x.into_iter().map(f64::to_bits).collect())
        };
        let (mut used, _) = solver(&first);
        let (mut fresh, _) = solver(&second);
        let _ = solved(&mut used, &first);
        let _ = solved(&mut used, &second);
        used.reset();
        prop_assert_eq!(used.statistics(), Work::default());
        prop_assert_eq!(solved(&mut used, &second), solved(&mut fresh, &second));
        prop_assert_eq!(used.statistics(), fresh.statistics());
        prop_assert_eq!(used.entries(), fresh.entries());
    }

    /// Verifies that cloned solvers, equivalent pivot states, and shared-view solves with
    /// separate workspaces all produce bit-identical results.
    #[test]
    fn a_clone_and_a_solver_alike_factor_alike((first, second) in reshuffled_pair()) {
        let (_, entries) = first.pattern();
        let solved = |lu: &mut SparseLu<f64>, matrix: &Matrix<f64>| {
            let stored = matrix.stored(lu.pattern(), &entries);
            let mut x: Vec<f64> = right_hand_side(matrix.dimension);
            let mut factors = lu.factor(&stored)?;
            let mut shared = x.clone();
            factors.solve(&mut x)?;
            let mut workspace = Workspace::new(matrix.dimension);
            factors.shared().solve(&mut shared, &mut workspace)?;
            assert_eq!(bits(&x), bits(&shared), "a shared solve solves alike");
            Ok::<Vec<u64>, LinearError>(bits(&x))
        };
        let (mut original, _) = solver(&first);
        prop_assert!(original.factors_alike(&original.clone()));
        let _ = solved(&mut original, &first);
        let mut clone = original.clone();
        prop_assert!(clone.factors_alike(&original));
        prop_assert_eq!(solved(&mut clone, &second), solved(&mut original, &second));
        prop_assert!(clone.factors_alike(&original));
        // A solver made afresh for the same ordering keeps no pivots, unlike one that has
        // factored; once it has factored the same matrices, it keeps the same pivots and
        // factors alike.
        let mut fresh = SparseLu::new(Arc::clone(original.ordering()));
        prop_assert!(!fresh.factors_alike(&original));
        let _ = solved(&mut fresh, &first);
        let _ = solved(&mut fresh, &second);
        prop_assert!(fresh.factors_alike(&original));
        prop_assert_eq!(solved(&mut fresh, &first), solved(&mut original, &first));
    }

    /// Verifies condensation of Meyer displacements across real, complex, dissected, and
    /// refactored solves against the dense reference.
    #[test]
    fn condenses_the_displacements_of_circuits(
        circuit in meyer(),
        alpha in -1.0..3.0f64,
        omega in -1.0..3.0f64,
    ) {
        let first = circuit.matrix(10f64.powf(alpha), false);
        let mut lu = condensing(&circuit, &first, None);
        let kept = first.dimension - circuit.displacements.len();
        prop_assert_eq!(lu.ordering().factored().dimension(), kept);
        check(&mut lu, &first)?;
        check(&mut lu, &circuit.matrix(10f64.powf(alpha), true))?;
        let statistics = lu.statistics();
        prop_assert_eq!(statistics.factorizations, 1);
        let complex = circuit.matrix(Complex64::new(0.0, 10f64.powf(omega)), false);
        check(&mut condensing(&circuit, &complex, Some(4)), &complex)?;
    }

    /// Verifies joint condensation of Meyer displacements and series-resistance currents across
    /// real, complex, dissected, and refactored solves.
    #[test]
    fn condenses_the_series_currents_of_circuits(
        circuit in switched(),
        alpha in -1.0..3.0f64,
        omega in -1.0..3.0f64,
    ) {
        let first = circuit.matrix(10f64.powf(alpha), false);
        let (pattern, _) = first.pattern();
        let condensation = Condensation::new(pattern, &circuit.pivots());
        let least = circuit.displacements.len() + circuit.series.len();
        prop_assert!(condensation.eliminated() >= least);
        let mut lu = condensing(&circuit, &first, None);
        let kept = first.dimension - condensation.eliminated();
        prop_assert_eq!(lu.ordering().factored().dimension(), kept);
        check(&mut lu, &first)?;
        check(&mut lu, &circuit.matrix(10f64.powf(alpha), true))?;
        let complex = circuit.matrix(Complex64::new(0.0, 10f64.powf(omega)), false);
        check(&mut condensing(&circuit, &complex, Some(4)), &complex)?;
    }

    /// Verifies that real and complex solvers share a single [`Ordering`] instance.
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

/// Converts each `f64` in `values` into its raw `u64` bit pattern.
fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Builds a dense 2 × 2 test [`Matrix`], its [`SparseLu`] solver, and slot-indexed values.
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

/// Factors and solves `matrix * x = [1, 2]` with `lu`.
fn solution(lu: &mut SparseLu<f64>, matrix: &Matrix<f64>) -> Result<Vec<f64>, LinearError> {
    let (_, entries) = matrix.pattern();
    let stored = matrix.stored(lu.pattern(), &entries);
    let mut x = vec![1.0, 2.0];
    lu.factor(&stored)?.solve(&mut x)?;
    Ok(x)
}

#[test]
fn falls_back_to_a_search_when_a_kept_pivot_becomes_too_small() {
    // The first matrix pivots on its diagonal. In the second matrix, each diagonal entry is
    // 10⁻¹⁴ of its off-diagonal entry, causing refactorization to fall back to a full search on
    // the first column.
    let (mut lu, _, first) = two_by_two([[4.0, 1.0], [1.0, 3.0]]);
    let x = solution(&mut lu, &first).unwrap();
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-15 && (x[1] - 7.0 / 11.0).abs() < 1e-15);
    // By Cramer's rule, [[a, b], [c, d]] x = [1, 2] has x = [d - 2b, 2a - c] / (ad - bc).
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
        Work {
            factorizations: 1,
            refactorizations: 0,
            fallbacks: 1,
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

/// Returns `true` when `result` is `Err(LinearError::Singular { .. })`.
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
        Work {
            factorizations: 1,
            refactorizations: 1,
            fallbacks: 0,
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

/// Builds a `stages`-stage resistive ladder where every node connects to a voltage-driven
/// supply rail and ground.
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
        displacements: Vec::new(),
        series: Vec::new(),
        changes: vec![1.0],
    };
    circuit.matrix(1.0, false)
}

#[test]
fn a_supply_joining_every_node_fills_nothing() {
    // The supply node and its branch current form singleton blocks, keeping supply entries above
    // the tridiagonal ladder block so elimination creates zero fill.
    let matrix = supplied_ladder(200);
    let (mut lu, entries) = solver(&matrix);
    assert_eq!(lu.ordering().blocks(), 3);
    let stored = matrix.stored(lu.pattern(), &entries);
    let mut x = right_hand_side::<f64>(matrix.dimension);
    let b = x.clone();
    lu.factor(&stored).unwrap().solve(&mut x).unwrap();
    assert!(backward_error(&matrix.dense(), &x, &b) < 1e-15);
    assert_eq!(lu.entries(), lu.pattern().len());
}

/// Builds a dissected [`SparseLu`] solver with at most `interiors` positions per interior and
/// returns its entry handles.
fn dissected<T: Scalar>(matrix: &Matrix<T>, interiors: usize) -> (SparseLu<T>, Vec<Entry>) {
    let (pattern, entries) = matrix.pattern();
    let ordering = Ordering::dissected(pattern, Dissecting::Every(interiors)).expect("fits");
    (SparseLu::new(Arc::new(ordering)), entries)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Verifies that dissected solvers pivot locally or delay pivots on row-shuffled dominant
    /// matrices within the `1e-10` backward error bound.
    #[test]
    fn dissected_solves_matrices_whose_large_entries_are_off_the_diagonal(
        matrix in shuffled_dominant(),
        interiors in 1usize..8,
    ) {
        let (mut lu, _) = dissected(&matrix, interiors);
        check_within(&mut lu, &matrix, 1e-10)?;
    }

    /// Verifies that dissected solvers match the serial backward error bound on unstructured
    /// random sparse matrices.
    #[test]
    fn dissected_solves_random_sparse_matrices(matrix in random_sparse(), interiors in 1usize..8) {
        let (mut lu, _) = dissected(&matrix, interiors);
        check(&mut lu, &matrix)?;
    }

    /// Verifies that dissected solvers factor and refactor real and complex circuit matrices,
    /// re-searching nodes whose pivots fail threshold checks.
    #[test]
    fn dissected_solves_and_refactors_the_equations_of_circuits(
        circuit in circuit(),
        alpha in -1.0..3.0f64,
        interiors in 1usize..8,
    ) {
        let first = circuit.matrix(10f64.powf(alpha), false);
        let second = circuit.matrix(10f64.powf(alpha), true);
        let (mut lu, _) = dissected(&first, interiors);
        check(&mut lu, &first)?;
        check(&mut lu, &second)?;
        let statistics = lu.statistics();
        prop_assert_eq!(statistics.factorizations, 1);
        prop_assert_eq!(statistics.refactorizations + statistics.fallbacks, 1);
        let complex = circuit.matrix(Complex64::new(0.0, 10f64.powf(alpha)), true);
        let (mut lu, _) = dissected(&complex, interiors);
        check(&mut lu, &complex)?;
    }

    /// Verifies that dissected refactorization searches afresh when cached pivots drop below
    /// threshold.
    #[test]
    fn dissected_searches_afresh_from_a_kept_pivot_too_small(
        (first, second) in reshuffled_pair(),
        interiors in 1usize..8,
    ) {
        let (mut lu, _) = dissected(&first, interiors);
        check(&mut lu, &first)?;
        check(&mut lu, &second)?;
        prop_assert_eq!(lu.statistics().factorizations, 1);
    }

    /// Verifies that dissected and serial solutions agree within `2e-12 * condition`.
    ///
    /// By Higham ("Accuracy and Stability of Numerical Algorithms", 2nd edition, 2002, Theorem
    /// 7.2), a backward error of ε yields a forward error within κ ε to first order.
    #[test]
    fn dissected_agrees_with_the_serial_solver(circuit in circuit(), interiors in 1usize..8) {
        let matrix = circuit.matrix(1.0, false);
        let dense = matrix.dense();
        let inverted = inverse(&dense).ok_or_else(|| TestCaseError::reject("singular"))?;
        let condition = norm(&dense) * norm(&inverted);
        prop_assume!(condition < 1e10, "too ill-conditioned for a sharp check");
        let solve = |mut lu: SparseLu<f64>, entries: Vec<Entry>| {
            let stored = matrix.stored(lu.pattern(), &entries);
            let mut x: Vec<f64> = right_hand_side(matrix.dimension);
            lu.factor(&stored).expect("regular").solve(&mut x).expect("finite");
            x
        };
        let (serial, entries) = solver(&matrix);
        let serial = solve(serial, entries);
        let (partitioned, entries) = dissected(&matrix, interiors);
        let partitioned = solve(partitioned, entries);
        let apart: Vec<f64> = serial.iter().zip(&partitioned).map(|(a, b)| a - b).collect();
        let distance = largest(&apart) / largest(&serial);
        prop_assert!(distance <= 2e-12 * condition.max(1.0), "{distance:e} apart, of condition {condition:e}");
    }

    /// Verifies that dissected factors, forward solves, and transposed solves produce
    /// bit-identical output on 1, 2, 4, and 8 threads.
    #[test]
    fn dissected_is_the_same_bits_on_any_number_of_threads(
        (first, second) in reshuffled_pair(),
        circuit in circuit(),
        interiors in 1usize..4,
    ) {
        let runs = |matrices: &[Matrix<f64>]| {
            let (mut lu, entries) = dissected(&matrices[0], interiors);
            lu.spread(rayon::current_num_threads());
            let mut found = Vec::new();
            for matrix in matrices {
                let stored = matrix.stored(lu.pattern(), &entries);
                let b: Vec<f64> = right_hand_side(matrix.dimension);
                let (mut x, mut y) = (b.clone(), b);
                let solved = lu.factor(&stored).and_then(|mut factors| {
                    factors.solve(&mut x)?;
                    factors.solve_transpose(&mut y)
                });
                found.push((solved.is_ok(), bits(&x), bits(&y), lu.delays()));
            }
            (found, lu.statistics())
        };
        let changed = (circuit.matrix(1.0, false), circuit.matrix(1.0, true));
        let cases = [vec![first, second], vec![changed.0.clone(), changed.1, changed.0]];
        for matrices in &cases {
            let alone = rayon::ThreadPoolBuilder::new().num_threads(1).build().expect("a pool");
            let once = alone.install(|| runs(matrices));
            for threads in [2, 4, 8] {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("a pool");
                prop_assert_eq!(&pool.install(|| runs(matrices)), &once, "on {} threads", threads);
            }
        }
    }
}

/// Builds a `side` × `side` resistive mesh matrix with ground conductances at every node.
fn mesh(side: usize, scale: f64) -> Matrix<f64> {
    let mut matrix = Matrix {
        dimension: side * side,
        positions: Vec::new(),
        values: Vec::new(),
    };
    let node = |x: usize, y: usize| x + side * y;
    for y in 0..side {
        for x in 0..side {
            let here = node(x, y);
            let ground = 1e-3 * f64::from(1 + u8::try_from(here % 7).expect("small"));
            matrix.positions.push((here, here));
            matrix.values.push(ground);
            let mut neighbors = Vec::new();
            if x + 1 < side {
                neighbors.push(node(x + 1, y));
            }
            if y + 1 < side {
                neighbors.push(node(x, y + 1));
            }
            for there in neighbors {
                let conductance = scale
                    * f64::from(1 + u8::try_from((here * 31 + there * 17) % 13).expect("small"));
                for (row, column, value) in [
                    (here, here, conductance),
                    (there, there, conductance),
                    (here, there, -conductance),
                    (there, here, -conductance),
                ] {
                    matrix.positions.push((row, column));
                    matrix.values.push(value);
                }
            }
        }
    }
    matrix
}

#[test]
fn a_dissected_mesh_is_the_same_bits_on_any_number_of_threads() {
    // A mesh of 1,600 nodes dissected into interiors of at most 40: dozens of interiors
    // and separators, factored, refactored and solved on 1, 2, 3, 4 and 8 threads, the
    // subtrees tasks that any of the threads takes.
    let first = mesh(40, 1.0);
    let second = mesh(40, 3.0);
    let run = || {
        let (mut lu, entries) = dissected(&first, 40);
        lu.spread(rayon::current_num_threads());
        let mut found = Vec::new();
        for matrix in [&first, &second, &first] {
            let stored = matrix.stored(lu.pattern(), &entries);
            let b: Vec<f64> = right_hand_side(matrix.dimension);
            let (mut x, mut y) = (b.clone(), b.clone());
            let mut factors = lu.factor(&stored).unwrap();
            factors.solve(&mut x).unwrap();
            factors.solve_transpose(&mut y).unwrap();
            assert!(backward_error(&matrix.dense(), &x, &b) < 1e-14);
            found.push((bits(&x), bits(&y)));
        }
        (found, lu.statistics(), lu.entries())
    };
    let alone = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    let once = alone.install(run);
    assert_eq!(once.1.factorizations, 1);
    assert_eq!(once.1.refactorizations + once.1.fallbacks, 2);
    for threads in [2, 3, 4, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(pool.install(run), once, "on {threads} threads");
    }
}

/// Builds a `side` × `side` mesh matrix coupled to a voltage-driven supply rail at every
/// `every`-th node.
fn supplied(side: usize, every: usize) -> Matrix<f64> {
    let mut matrix = mesh(side, 1.0);
    let nodes = side * side;
    let (supply, current) = (nodes, nodes + 1);
    matrix.dimension = nodes + 2;
    for node in (0..nodes).step_by(every) {
        for (row, column, value) in [
            (node, node, 0.5),
            (supply, supply, 0.5),
            (node, supply, -0.5),
            (supply, node, -0.5),
        ] {
            matrix.positions.push((row, column));
            matrix.values.push(value);
        }
    }
    for (row, column) in [(supply, current), (current, supply)] {
        matrix.positions.push((row, column));
        matrix.values.push(1.0);
    }
    matrix
}

#[test]
fn a_supplied_mesh_is_the_same_bits_on_any_number_of_threads() {
    // Test a 2,304-node mesh with a supply rail connected to 768 nodes and then all 2,304
    // nodes. The supply rail creates multi-chunk off-diagonal row spans tested both whole and
    // dissected across 1, 2, 3, and 8 threads.
    for every in [3, 1] {
        same_bits_on_any_number_of_threads(&supplied(48, every));
    }
}

/// Asserts that `matrix`, solved whole and dissected into interiors of at most 40, and
/// solved transposed, gives the same bits on 1, 2, 3 and 8 threads.
fn same_bits_on_any_number_of_threads(matrix: &Matrix<f64>) {
    let dense = matrix.dense();
    let b: Vec<f64> = right_hand_side(matrix.dimension);
    let run = |interiors: Option<usize>| {
        let (mut lu, entries) = match interiors {
            Some(most) => dissected(matrix, most),
            None => solver(matrix),
        };
        assert!(lu.ordering().blocks() >= 3, "the supply's blocks apart");
        lu.spread(rayon::current_num_threads());
        let stored = matrix.stored(lu.pattern(), &entries);
        let (mut x, mut y) = (b.clone(), b.clone());
        let mut factors = lu.factor(&stored).unwrap();
        factors.solve(&mut x).unwrap();
        factors.solve_transpose(&mut y).unwrap();
        assert!(backward_error(&dense, &x, &b) < 1e-14);
        (bits(&x), bits(&y))
    };
    let pool = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    };
    for interiors in [None, Some(40)] {
        let once = pool(1).install(|| run(interiors));
        for threads in [2, 3, 8] {
            let found = pool(threads).install(|| run(interiors));
            assert_eq!(found, once, "{interiors:?}, on {threads} threads");
        }
    }
}

#[test]
fn delays_a_pivot_its_interior_cannot_take() {
    // Dissect a three-node path into single-node interiors with node 1 as separator. Column 0's
    // diagonal entry is 10⁻⁶ of its separator entry in row 1, forcing the interior to delay the
    // column to the separator where it pivots accurately.
    let matrix = Matrix {
        dimension: 3,
        positions: vec![(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)],
        values: vec![1e-6, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0],
    };
    let (mut lu, _) = dissected(&matrix, 1);
    check(&mut lu, &matrix).unwrap();
    assert_eq!(lu.delays(), 1);
    // The same matrix refactors with the delay kept.
    check(&mut lu, &matrix).unwrap();
    assert_eq!(lu.delays(), 1);
}

#[test]
fn delays_pivots_on_many_random_matrices() {
    // Over a thousand shuffled dominant matrices dissected into small interiors, the
    // property test above solves some whose pivots its interiors delay.
    use proptest::strategy::ValueTree;
    let mut runner = proptest::test_runner::TestRunner::deterministic();
    let mut delayed = 0;
    for _ in 0..1000 {
        let matrix = shuffled_dominant().new_tree(&mut runner).unwrap().current();
        let (mut lu, entries) = dissected(&matrix, 2);
        let stored = matrix.stored(lu.pattern(), &entries);
        if lu.factor(&stored).is_ok() && lu.delays() > 0 {
            delayed += 1;
        }
    }
    assert!(delayed > 10, "{delayed} of a thousand delayed a pivot");
}

#[test]
fn reports_a_singular_dissected_matrix() {
    // The mesh with its last column zero: the root separator finds no pivot for it.
    let mut matrix = mesh(10, 1.0);
    for (position, value) in matrix.positions.iter().zip(matrix.values.iter_mut()) {
        if position.1 == 99 {
            *value = 0.0;
        }
    }
    let (mut lu, entries) = dissected(&matrix, 5);
    let stored = matrix.stored(lu.pattern(), &entries);
    assert_eq!(
        lu.factor(&stored).err(),
        Some(LinearError::Singular { columns: vec![99] })
    );
}
