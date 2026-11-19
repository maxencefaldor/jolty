//! Poles, zeros, and growing natural modes of the circuit linearized at its DC operating
//! point.
//!
//! At complex frequency `s`, the linearized circuit satisfies `(G + sC) x = -u * b`, where
//! `u` is an input source level and `b` is its residual derivative. Poles are the finite
//! generalized eigenvalues of `(G, -C)` where `G + sC` becomes singular. For an output
//! selector `e`, the transfer function `H(s) = -eᵀ (G + sC)⁻¹ b` vanishes where the bordered
//! pencil `[[G + sC, b], [eᵀ, 0]]` is singular, yielding the finite zeros. Uncontrollable or
//! unobservable natural frequencies appear in both lists.
//!
//! Dense QZ solves compute eigenvalues after scaling frequency by `||G|| / ||C||` and
//! equilibrating rows and columns to unit scale. Eigenvalues above `10¹²` times the
//! characteristic frequency are classified as infinite, and zeros within `10⁻¹⁰` of the
//! largest root snap to the origin. Real pencils yield roots snapped to exact real values or
//! exact complex conjugate pairs.
//!
//! Poles with positive real parts correspond to growing natural modes such as oscillating
//! loops or tipping latches. Mode shapes solve `(G + sC) x = 0` via shifted sparse inverse
//! iteration near the pole from pseudo-random seeds (as in LAPACK `xLARNV` / `xSTEIN`).
//! Random seeds preserve nonzero projections onto symmetric modes, such as ring-oscillator
//! stages, where uniform seeds vanish.
//!
//! Dense pencil memory grows quadratically with unknown count, so analyses validate memory
//! bounds against `options.max_memory` before allocating.

use std::cmp::Ordering;

use jolty_numerics::{Complex64, Eigenvalue, PencilError, generalized_eigenvalues};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::Deserialize;

use crate::ac::selector_vector;
use crate::{OperatingPointResult, Options, SimulationError, Simulator};

/// Ratio `|beta| / |alpha|` below which a generalized eigenvalue is classified as infinite
/// (beyond `10¹²` times the circuit's characteristic frequency).
///
/// Derivation: QZ rounding threshold (`~4.5e3 * f64::EPSILON`) above `O(n * eps)` backward
/// error on algebraic infinite eigenvalues of the equilibrated pencil `(G, -C)`.
const INFINITE: f64 = 1e-12;

/// Fraction of the largest finite root below which a zero snaps to the origin.
///
/// Derivation: `100 * INFINITE` threshold absorbing Newton operating-point and QZ
/// cancellation rounding on exact DC zeros (such as coupling capacitors) without discarding
/// low-frequency finite zeros.
const ORIGIN: f64 = 1e-10;

/// Magnitude threshold below which both `alpha` and `beta` vanish, signaling a singular
/// pencil with an identically zero determinant.
///
/// Derivation: QZ singular-pencil rounding threshold (`~4.5e3 * f64::EPSILON`, matching
/// `INFINITE`) on the unit-norm equilibrated pencil.
const DEGENERATE: f64 = 1e-12;

/// Relative imaginary-part threshold below which a root of a real pencil is snapped to real.
///
/// Derivation: `~4.5e6 * f64::EPSILON` (three decades above `INFINITE`), removing QZ
/// imaginary rounding noise from real roots while keeping lightly damped conjugate pairs.
const REAL: f64 = 1e-9;

/// Number of Ruiz equilibration passes scaling row and column infinity norms toward unity.
///
/// Source: D. Ruiz, "A scaling algorithm to equilibrate both rows and columns norms in
/// matrices", Technical Report `RAL-TR-2001-034`, 2001 (quadratic convergence; 4 passes
/// balance typical `G`/`C` pencils).
const PASSES: usize = 4;

/// Minimum relative real part (`Re(s) / |s|`) required to classify a pole as growing.
///
/// Derivation: stability threshold (`zeta < -1e-6`, three decades above `REAL`) preventing
/// QZ rounding of lossless LC resonator poles from triggering spurious growth.
const GROWING: f64 = 1e-6;

/// Relative imaginary frequency shift applied to a pole before sparse inverse iteration.
///
/// Source: shifted inverse iteration (Golub and Van Loan, *Matrix Computations*, 4th ed.,
/// 2013); `1e-6` amplifies the target mode by `~1e6` per solve while keeping `G + sC`
/// conditioned to ~10 digits in `f64`.
const PERTURBATION: f64 = 1e-6;

/// Number of shifted inverse-iteration solves per growing mode.
///
/// Derivation: two solves at shift `1e-6` suppress competing modes by `~1e12`, matching `f64`
/// accuracy at the shifted matrix's `~1e6` conditioning.
const ITERATIONS: usize = 2;

/// Peak number of dense real `n × n` matrices live during a pole-zero solve (or
/// `(n + 1) × (n + 1)` when bordered for zeros).
///
/// Derivation: exact count `1 (G) + 1 (C) + 2 (equilibrated) + 2 * 2 (complex QZ) = 8`.
const PENCIL_MATRICES: f64 = 8.0;
/// Analysis label used in pole-zero diagnostic messages.
const POLE_ZERO: &str = "pole-zero analysis";
/// Refusal message for circuits with delay lines, whose `e^{-sτ}` factors make `det(G + sC)`
/// transcendental with infinitely many poles.
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

/// Natural mode of the circuit linearized at its operating point.
#[derive(Debug, Clone, PartialEq)]
pub struct Mode {
    /// Mode pole in radians per second; for a conjugate pair, the member with non-negative
    /// imaginary part.
    pub pole: Complex64,
    /// Per-unknown complex mode amplitude normalized to unit maximum magnitude.
    pub shape: Vec<Complex64>,
}

impl Simulator {
    /// Linearizes the circuit at its operating point and computes its poles, plus transfer
    /// function zeros when both a source and an output are specified.
    ///
    /// # Errors
    ///
    /// Returns [`SimulationError::InvalidAnalysis`] for incomplete source/output pairs,
    /// invalid references, or identically zero transfer functions. Other failure variants
    /// include [`SimulationError::UnknownOutput`], [`SimulationError::UnknownSource`],
    /// [`SimulationError::NotASource`], [`SimulationError::Singular`],
    /// [`SimulationError::Unconverged`], [`SimulationError::TooMuchMemory`], and errors from
    /// [`Simulator::operating_point`].
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

    /// Computes growing natural modes (`Re(s) > 1e-6 * |s|`) ordered by decreasing growth rate,
    /// listing each conjugate pair once.
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
                    let shifted = pole * Complex64::new(1.0, PERTURBATION);
                    let mut factors =
                        linearization.factor_at(&self.state.workspace.contributions, shifted)?;
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

    /// Copies workspace matrices `G` and `C` into dense column-major buffers.
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

/// Builds bordered `(n + 1) × (n + 1)` column-major pencil matrices `[[g, level], [readᵀ, 0]]`
/// and `[[c, 0], [0, 0]]`.
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

/// Constructs a [`SimulationError::Singular`] error for a pencil singular at every frequency.
fn everywhere() -> SimulationError {
    SimulationError::Singular {
        unknowns: Vec::new(),
    }
}

/// Computes sorted finite roots of `det(g + s * c)` for `n × n` column-major matrices, or
/// returns `None` if the pencil is identically singular.
fn roots(
    n: usize,
    mut g: Vec<f64>,
    mut c: Vec<f64>,
) -> Result<Option<Vec<Complex64>>, SimulationError> {
    let norm = |matrix: &[f64]| matrix.iter().map(|value| value * value).sum::<f64>().sqrt();
    let (g_norm, c_norm) = (norm(&g), norm(&c));
    // Characteristic frequency scaling the pencil roots to order 1.
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
    // Identically singular pencils produce `0 / 0` generalized eigenvalues.
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

/// Snaps `roots` of a real pencil to exact real values (within [`REAL`]) or averaged exact
/// complex conjugate pairs.
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

/// Applies Ruiz row-and-column equilibration to the `n × n` column-major pencil `(g, c)` so
/// row and column infinity norms approach 1 without changing eigenvalues.
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
