//! The poles and zeros of the circuit linearized at its operating point.
//!
//! Linearized, the circuit's equations are `(G + sC) x = −u · b` at the complex frequency
//! `s`, where `u` is an input source's level and `b` the derivative of the residual with
//! respect to it. Its poles, its natural frequencies, are the `s` at which `G + sC` is singular: the
//! finite generalized eigenvalues of the pencil `(G, −C)`. The transfer function from the
//! source to an output read by `e`, `H(s) = −eᵀ (G + sC)⁻¹ b`, vanishes where the bordered
//! matrix `[[G + sC, b], [eᵀ, 0]]` is singular, so its zeros are the finite eigenvalues of
//! the bordered pencil. A natural frequency that the source cannot excite, or the output
//! cannot see, is a root of both, and both list it.
//!
//! Both pencils are dense and solved by the QZ algorithm, after scaling the frequency by
//! the circuit's own, `‖G‖ / ‖C‖`, and equilibrating rows and columns, so that their
//! entries are of one order; an eigenvalue beyond `10¹²` times that frequency is infinite.
//! The circuit has an operating point, so `G` is regular and no pole lies at the origin;
//! a zero within `10⁻¹⁰` times the largest pole or zero lies there, as a coupling
//! capacitor's does. The pencils are real, so their roots are real or come in conjugate
//! pairs, which the roots found are made exactly.
//!
//! A pole with a positive real part is a natural mode that grows away from the operating
//! point: an oscillation that builds up, or a latch that tips. Its shape, the amplitude of
//! each unknown in it, solves `(G + sC) x = 0` at the pole, found by inverse iteration
//! with the sparse factorization of `G + sC` just beside it, from pseudo-random
//! amplitudes, as LAPACK's inverse iteration starts (`xLARNV` in `xSTEIN`): a start of
//! equal amplitudes has no part along a mode of a circuit that a permutation of its
//! unknowns leaves as it is, such as a ring oscillator's, whose stages swing in turn.
//!
//! The dense pencils take memory as the square of the unknowns, so an analysis checks
//! them against its options' `max_memory` before it makes them, and fails if they pass it.

use std::cmp::Ordering;

use jolty_numerics::{Complex64, Eigenvalue, PencilError, generalized_eigenvalues};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::Deserialize;

use crate::ac::selector_vector;
use crate::{OperatingPointResult, Options, SimulationError, Simulator};

/// The ratio of an eigenvalue's denominator to its numerator below which it is infinite: a
/// root beyond `10¹²` times the circuit's characteristic frequency.
///
/// Origin: jolty's own QZ rounding threshold (`~4.5e3 * f64::EPSILON`), above typical
/// `O(n * eps)` backward error of the QZ factorization on algebraic (infinite) eigenvalues
/// of the scaled, equilibrated pencil `(G, -C)`.
const INFINITE: f64 = 1e-12;

/// The fraction of the largest root below which a zero is at the origin: no closer than the
/// linearized circuit, solved to Newton's tolerance, resolves it.
///
/// Origin: jolty's own QZ origin-snapping threshold (`100 * INFINITE`), wide enough to absorb
/// Newton operating-point and QZ cancellation noise on exact DC zeros (such as coupling
/// capacitors) while preserving low-frequency finite zeros.
const ORIGIN: f64 = 1e-10;

/// The size below which both parts of an eigenvalue of the equilibrated pencil vanish, as
/// they do for a pencil whose determinant is zero everywhere.
///
/// Origin: jolty's own QZ singular-pencil threshold (`~4.5e3 * f64::EPSILON`, matching
/// `INFINITE`) on the equilibrated pencil whose entries have norm near `1`, detecting `0 / 0`
/// eigenvalues.
const DEGENERATE: f64 = 1e-12;

/// The imaginary part, relative to its magnitude, below which a root is real: the error of
/// the complex QZ iteration on a real pencil.
///
/// Origin: jolty's own threshold (`~4.5e6 * f64::EPSILON`, three decades above `INFINITE`),
/// stripping spurious QZ imaginary noise from real roots while retaining lightly damped
/// conjugate pairs.
const REAL: f64 = 1e-9;

/// The equilibration passes, each of which brings every row's and column's largest entry
/// nearer to 1.
///
/// Source: Ruiz equilibration (D. Ruiz, "A scaling algorithm to equilibrate both rows and
/// columns norms in matrices", Technical Report `RAL-TR-2001-034`, 2001), which converges
/// quadratically in the infinity norm; 4 passes is jolty's fixed iteration count, sufficient
/// to bring typical `G`/`C` imbalances close to unit row and column norms.
const PASSES: usize = 4;

/// The real part, relative to its magnitude, above which a pole grows: well above the
/// error of the QZ iteration, so that a lossless resonator's poles do not.
///
/// Origin: jolty's own stability threshold (equivalent to negative damping ratio `zeta < -1e-6`,
/// three decades above `REAL`), keeping numerical perturbation of imaginary-axis LC resonator
/// poles from triggering false positive growth.
const GROWING: f64 = 1e-6;

/// The distance from a pole, relative to its magnitude, at which inverse iteration factors
/// `G + sC`: near enough that the pole's mode dominates the solution a million times over,
/// far enough that the factorization stays regular.
///
/// Source: shifted inverse iteration (Golub and Van Loan, *Matrix Computations*, 4th ed.,
/// 2013); jolty's relative imaginary shift `1e-6` amplifies the target mode by `~1e6` per
/// solve while leaving `G + sC` conditioned by `~1e6` (~10 digits preserved in `f64`).
const BESIDE: f64 = 1e-6;

/// The steps of inverse iteration, each of which multiplies the mode's lead by `1 / BESIDE`.
///
/// Derivation: with `BESIDE = 1e-6`, two solves suppress competing modes by `~1 / BESIDE^2 =
/// 1e12` (near `f64` accuracy relative to the shifted matrix conditioning `~1e6`).
const ITERATIONS: usize = 2;

/// The most dense real matrices a pole-zero analysis holds at once, each of a row per
/// unknown, and one more where it borders them for the zeros: the circuit's `G` and `C`,
/// and, for the pencil it solves, the two it equilibrates and the two complex ones its QZ
/// iteration works on, each the size of two real ones.
///
/// Derivation: exact count `1 (G) + 1 (C) + 2 (equilibrated) + 2 * 2 (complex QZ) = 8`.
const PENCIL_MATRICES: f64 = 8.0;
/// The analysis's name in its messages.
/// Origin: shared jolty diagnostic label (no numeric parameter).
const POLE_ZERO: &str = "pole-zero analysis";
/// Why a pole-zero analysis, and the search for growing modes, take no delay line: its
/// `e^{−sτ}` makes the circuit's determinant transcendental in `s`, whose roots are
/// infinitely many, where the pencil `G + sC` has as many as its unknowns at most.
const INFINITELY_MANY_POLES: &str = "a delay gives a circuit infinitely many poles";

/// A pole-zero analysis: the poles of the circuit linearized at its operating point, and,
/// given a source and an output, the zeros of the transfer function between them.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoleZero {
    /// The path of the independent source the transfer function is from, such as `V1`.
    #[serde(default)]
    pub source: Option<String>,
    /// The output the transfer function is to, an unknown by name, such as `out`.
    #[serde(default)]
    pub output: Option<String>,
    /// The node the output voltage is taken over; by default ground.
    #[serde(default)]
    pub reference: Option<String>,
}

/// The poles and zeros of the circuit linearized at its operating point.
#[derive(Debug, Clone, PartialEq)]
pub struct PoleZeroResult {
    /// The path of the source the transfer function is from, if any.
    pub source: Option<String>,
    /// The output the transfer function is to, if any.
    pub output: Option<String>,
    /// The node the output voltage is taken over, if not ground.
    pub reference: Option<String>,
    /// The operating point the circuit was linearized at.
    pub operating_point: OperatingPointResult,
    /// The poles, complex frequencies in radians per second: the finite roots of
    /// `det(G + sC)`, by increasing magnitude, a conjugate pair's positive member first.
    pub poles: Vec<Complex64>,
    /// The zeros of the transfer function from the source to the output, in the same
    /// order, given both.
    pub zeros: Option<Vec<Complex64>>,
}

/// A natural mode of the circuit linearized at its operating point.
#[derive(Debug, Clone, PartialEq)]
pub struct Mode {
    /// Its pole, in radians per second; of a conjugate pair, the member above the real
    /// axis.
    pub pole: Complex64,
    /// The amplitude of each unknown in the mode, the largest of magnitude 1.
    pub shape: Vec<Complex64>,
}

impl Simulator {
    /// Linearizes the circuit at its operating point and finds its poles, and the zeros
    /// of the transfer function from the source to the output, given both.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for a source without an output, an
    /// output or a reference without a source, an output over itself or a current over a
    /// node, or a transfer function that is zero at every frequency;
    /// [`SimulationError::UnknownOutput`], [`SimulationError::UnknownSource`] or
    /// [`SimulationError::NotASource`] when they name no such unknown or source;
    /// [`SimulationError::Singular`] when the linearized circuit is singular at every
    /// frequency; [`SimulationError::Unconverged`] when the QZ iteration does not
    /// converge; [`SimulationError::TooMuchMemory`] when its dense matrices would hold
    /// more than the options' `max_memory`; and the errors of [`Simulator::operating_point`].
    pub fn pole_zero(
        &mut self,
        analysis: &PoleZero,
        options: &Options,
    ) -> Result<PoleZeroResult, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            let transfer = match (&analysis.source, &analysis.output) {
                (Some(source), Some(output)) => {
                    let index = self.source_index(source)?;
                    let (output, reference, _) =
                        self.measured(output, analysis.reference.as_deref())?;
                    Some((index, output, reference))
                }
                (None, None) if analysis.reference.is_none() => None,
                _ => {
                    return Err(SimulationError::InvalidAnalysis(
                        "a pole-zero analysis's zeros need a source and an output, both or neither"
                            .to_owned(),
                    ));
                }
            };
            options.check()?;
            self.undelayed(POLE_ZERO, INFINITELY_MANY_POLES)?;
            let rows = self.elaborated.unknowns.len() + usize::from(transfer.is_some());
            options.hold_square(POLE_ZERO, rows, PENCIL_MATRICES, 0)?;
            let operating_point = self.linearized(options)?;
            let n = self.elaborated.unknowns.len();
            let (conductance, capacitance) = self.dense_pencil();
            let (poles, zeros) = if let Some((index, output, reference)) = transfer {
                let poles =
                    roots(n, conductance.clone(), capacitance.clone())?.ok_or_else(everywhere)?;
                let level = self.source_excitation(index);
                let read: Vec<f64> = selector_vector(n, output, reference);
                let (g, c) = bordered(n, &conductance, &capacitance, &level, &read);
                drop(conductance);
                drop(capacitance);
                let zeros = roots(n + 1, g, c)?.ok_or_else(|| {
                    SimulationError::InvalidAnalysis(
                        "the transfer function from the source to the output is zero at every \
                         frequency, so it has no zeros"
                            .to_owned(),
                    )
                })?;
                let largest = poles.iter().chain(&zeros).map(|root| root.norm());
                let origin = ORIGIN * largest.fold(0.0, f64::max);
                let zeros = zeros
                    .into_iter()
                    .map(|zero| {
                        if zero.norm() <= origin {
                            Complex64::default()
                        } else {
                            zero
                        }
                    })
                    .collect();
                (poles, Some(zeros))
            } else {
                let poles = roots(n, conductance, capacitance)?.ok_or_else(everywhere)?;
                (poles, None)
            };
            Ok(PoleZeroResult {
                source: analysis.source.clone(),
                output: analysis.output.clone(),
                reference: analysis.reference.clone(),
                operating_point,
                poles,
                zeros,
            })
        })
    }

    /// The natural modes that grow away from the operating point: those whose poles have a
    /// real part above a millionth of their magnitude, a conjugate pair once, by
    /// decreasing real part.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`Simulator::pole_zero`] without a source.
    pub fn growing_modes(&mut self, options: &Options) -> Result<Vec<Mode>, SimulationError> {
        options.parallelism.confined(|| {
            self.solvable()?;
            options.check()?;
            self.undelayed("search for growing modes", INFINITELY_MANY_POLES)?;
            options.hold_square(
                POLE_ZERO,
                self.elaborated.unknowns.len(),
                PENCIL_MATRICES,
                0,
            )?;
            self.linearized(options)?;
            let n = self.elaborated.unknowns.len();
            let (conductance, capacitance) = self.dense_pencil();
            let poles = roots(n, conductance, capacitance)?.ok_or_else(everywhere)?;
            let mut growing: Vec<Complex64> = poles
                .into_iter()
                .filter(|pole| pole.re > GROWING * pole.norm() && pole.im >= 0.0)
                .collect();
            growing.sort_by(|x, y| y.re.total_cmp(&x.re));
            let mut linearization = self.linearization();
            growing
                .into_iter()
                .map(|pole| {
                    let beside = pole * Complex64::new(1.0, BESIDE);
                    let mut factors =
                        linearization.factor_at(&self.state.workspace.contributions, beside)?;
                    // Any seed serves: a draw has a part along every mode.
                    let mut draws = ChaCha20Rng::seed_from_u64(0);
                    let mut shape: Vec<Complex64> = (0..n)
                        .map(|_| Complex64::new(draws.random_range(-1.0..1.0), 0.0))
                        .collect();
                    for _ in 0..ITERATIONS {
                        factors.solve(&mut shape)?;
                        let largest = shape.iter().map(|value| value.norm()).fold(0.0, f64::max);
                        for value in &mut shape {
                            *value /= largest;
                        }
                    }
                    Ok(Mode { pole, shape })
                })
                .collect()
        })
    }

    /// The matrices `G` and `C` the workspace holds, dense in column-major order.
    fn dense_pencil(&self) -> (Vec<f64>, Vec<f64>) {
        let n = self.elaborated.unknowns.len();
        let mut conductance = vec![0.0; n * n];
        let mut capacitance = vec![0.0; n * n];
        let contributions = &self.state.workspace.contributions;
        for (((row, column), &g), &c) in self
            .elaborated
            .pattern
            .positions()
            .zip(&contributions.conductance)
            .zip(&contributions.capacitance)
        {
            conductance[column * n + row] += g;
            capacitance[column * n + row] += c;
        }
        (conductance, capacitance)
    }
}

/// The dense matrices `[[g, level], [readᵀ, 0]]` and `[[c, 0], [0, 0]]` of `n + 1` rows,
/// column-major, from `g` and `c` of `n`.
fn bordered(n: usize, g: &[f64], c: &[f64], level: &[f64], read: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let size = n + 1;
    let mut bordered_g = vec![0.0; size * size];
    let mut bordered_c = vec![0.0; size * size];
    for (index, (&x, &y)) in g.iter().zip(c).enumerate() {
        let (row, column) = (index % n, index / n);
        bordered_g[column * size + row] = x;
        bordered_c[column * size + row] = y;
    }
    for (index, (&reading, &excitation)) in read.iter().zip(level).enumerate() {
        bordered_g[index * size + n] = reading;
        bordered_g[n * size + index] = excitation;
    }
    (bordered_g, bordered_c)
}

/// The error of a linearized circuit whose `det(G + sC)` is zero at every `s`: singular,
/// with no unknown standing out.
fn everywhere() -> SimulationError {
    SimulationError::Singular {
        unknowns: Vec::new(),
    }
}

/// The finite roots of `det(g + s·c)`, of two dense matrices of `n` rows in column-major
/// order, sorted, or `None` when the determinant is zero at every `s`.
fn roots(
    n: usize,
    mut g: Vec<f64>,
    mut c: Vec<f64>,
) -> Result<Option<Vec<Complex64>>, SimulationError> {
    let norm = |matrix: &[f64]| matrix.iter().map(|value| value * value).sum::<f64>().sqrt();
    let (g_norm, c_norm) = (norm(&g), norm(&c));
    // The circuit's characteristic frequency, by which the roots are scaled to order 1.
    let frequency = if c_norm > 0.0 && g_norm > 0.0 {
        g_norm / c_norm
    } else {
        1.0
    };
    for value in &mut c {
        *value *= -frequency;
    }
    equilibrate(n, &mut g, &mut c);
    let eigenvalues = generalized_eigenvalues(n, &g, &c).map_err(|error| match error {
        PencilError::Unconverged => SimulationError::Unconverged,
    })?;
    // A pencil whose determinant vanishes everywhere has an eigenvalue 0/0.
    if eigenvalues
        .iter()
        .any(|Eigenvalue { alpha, beta }| alpha.norm() <= DEGENERATE && beta.norm() <= DEGENERATE)
    {
        return Ok(None);
    }
    let finite = eigenvalues
        .iter()
        .filter(|Eigenvalue { alpha, beta }| beta.norm() > INFINITE * alpha.norm())
        .map(|Eigenvalue { alpha, beta }| alpha / beta * frequency)
        .collect();
    let mut found = conjugated(finite);
    found.sort_by(|x, y| {
        x.norm()
            .partial_cmp(&y.norm())
            .unwrap_or(Ordering::Equal)
            .then(y.im.total_cmp(&x.im))
    });
    Ok(Some(found))
}

/// `roots` of a real pencil, made exactly real or exactly conjugate in pairs: the roots
/// within [`REAL`] of the real axis lose their imaginary parts, and each root above it is
/// averaged with the conjugate of the nearest below it.
fn conjugated(roots: Vec<Complex64>) -> Vec<Complex64> {
    let (mut real, mut upper, mut lower) = (Vec::new(), Vec::new(), Vec::new());
    for root in roots {
        if root.im.abs() <= REAL * root.norm() {
            real.push(Complex64::new(root.re, 0.0));
        } else if root.im > 0.0 {
            upper.push(root);
        } else {
            lower.push(root);
        }
    }
    let mut paired = real;
    for root in upper {
        let nearest = lower
            .iter()
            .enumerate()
            .min_by(|(_, x), (_, y)| {
                let distance = |other: &Complex64| (root - other.conj()).norm();
                distance(x).total_cmp(&distance(y))
            })
            .map(|(index, _)| index);
        match nearest {
            Some(index) => {
                let mean = (root + lower.swap_remove(index).conj()) / 2.0;
                paired.extend([mean, mean.conj()]);
            }
            None => paired.push(root),
        }
    }
    paired.extend(lower);
    paired
}

/// Scales the rows and columns of the pencil `(g, c)`, of `n` rows in column-major order,
/// so that each row's and column's largest entry nears 1, which leaves its eigenvalues as
/// they are (Ruiz's equilibration).
fn equilibrate(n: usize, g: &mut [f64], c: &mut [f64]) {
    let scale = |largest: f64| {
        if largest > 0.0 {
            largest.sqrt().recip()
        } else {
            1.0
        }
    };
    let mut rows = vec![0.0_f64; n];
    let mut columns = vec![0.0_f64; n];
    for _ in 0..PASSES {
        rows.fill(0.0);
        columns.fill(0.0);
        for (index, (x, y)) in g.iter().zip(c.iter()).enumerate() {
            let size = x.abs().max(y.abs());
            let (row, column) = (index % n, index / n);
            rows[row] = rows[row].max(size);
            columns[column] = columns[column].max(size);
        }
        for value in &mut rows {
            *value = scale(*value);
        }
        for value in &mut columns {
            *value = scale(*value);
        }
        for (index, (x, y)) in g.iter_mut().zip(c.iter_mut()).enumerate() {
            let factor = rows[index % n] * columns[index / n];
            *x *= factor;
            *y *= factor;
        }
    }
}
