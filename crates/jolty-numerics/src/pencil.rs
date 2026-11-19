//! The generalized eigenvalues of dense matrix pencils, by the QZ algorithm (Moler and
//! Stewart, "An algorithm for generalized matrix eigenvalue problems", SIAM Journal on
//! Numerical Analysis 10(2), 1973), in complex arithmetic, as LAPACK's ZGGHRD and ZHGEQZ
//! compute them.
//!
//! The pencil is first reduced by plane rotations to a Hessenberg matrix and a triangular
//! one, then each eigenvalue is found by single-shift QZ sweeps on the part not yet
//! deflated, the infinite ones, where the triangular matrix holds a zero on its diagonal,
//! deflated as they appear. Complex arithmetic gives each eigenvalue of a real pencil's
//! conjugate pair its own numerator and denominator, and every operation is the same on
//! every machine.

use num_complex::Complex64;

/// Additive identity of complex numbers (`0.0 + 0.0i`).
const ZERO: Complex64 = Complex64::new(0.0, 0.0);

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
    /// The QZ iteration reached its limit, thirty sweeps per row, as LAPACK's, before every
    /// eigenvalue converged.
    #[error("the QZ iteration did not converge")]
    Unconverged,
}

/// The generalized eigenvalues of the pencil `(a, b)`, the `λ` for which `a v = λ b v` with
/// `v` nonzero, of two dense square matrices of `dimension` rows, in column-major order.
///
/// # Errors
///
/// Returns [`PencilError::Unconverged`] when either matrix contains a non-finite entry or when
/// the QZ iteration does not converge.
///
/// # Panics
///
/// Panics unless each matrix holds `dimension²` values.
pub fn generalized_eigenvalues(
    dimension: usize,
    a: &[f64],
    b: &[f64],
) -> Result<Vec<Eigenvalue>, PencilError> {
    assert_eq!(
        a.len(),
        dimension * dimension,
        "a square matrix of the dimension"
    );
    assert_eq!(
        b.len(),
        dimension * dimension,
        "a square matrix of the dimension"
    );
    if a.iter().chain(b).any(|value| !value.is_finite()) {
        return Err(PencilError::Unconverged);
    }
    let mut h = Square::from_real(dimension, a);
    let mut t = Square::from_real(dimension, b);
    triangularize(&mut h, &mut t);
    reduce_to_hessenberg(&mut h, &mut t);
    if h.entries
        .iter()
        .chain(&t.entries)
        .any(|entry| !entry.re.is_finite() || !entry.im.is_finite())
    {
        return Err(PencilError::Unconverged);
    }
    qz(&mut h, &mut t)
}

/// A dense square complex matrix, column-major.
struct Square {
    n: usize,
    entries: Vec<Complex64>,
}

impl Square {
    fn from_real(n: usize, entries: &[f64]) -> Self {
        Self {
            n,
            entries: entries
                .iter()
                .map(|&value| Complex64::new(value, 0.0))
                .collect(),
        }
    }

    fn at(&self, row: usize, column: usize) -> Complex64 {
        self.entries[column * self.n + row]
    }

    fn set(&mut self, row: usize, column: usize, value: Complex64) {
        self.entries[column * self.n + row] = value;
    }

    /// Rotates rows `first` and `second` over `columns`: `(x, y)` becomes
    /// `(c x + s y, c y − s̄ x)`, as LAPACK's ZROT.
    fn rotate_rows(
        &mut self,
        (first, second): (usize, usize),
        columns: std::ops::Range<usize>,
        (c, s): (f64, Complex64),
    ) {
        for column in columns {
            let (x, y) = (self.at(first, column), self.at(second, column));
            self.set(first, column, c * x + s * y);
            self.set(second, column, c * y - s.conj() * x);
        }
    }

    /// Rotates columns `first` and `second` over `rows` likewise.
    fn rotate_columns(
        &mut self,
        (first, second): (usize, usize),
        rows: std::ops::Range<usize>,
        (c, s): (f64, Complex64),
    ) {
        for row in rows {
            let (x, y) = (self.at(row, first), self.at(row, second));
            self.set(row, first, c * x + s * y);
            self.set(row, second, c * y - s.conj() * x);
        }
    }

    /// The Frobenius norm of the matrix's upper Hessenberg part.
    fn hessenberg_norm(&self) -> f64 {
        let mut sum = 0.0;
        for column in 0..self.n {
            for row in 0..self.n.min(column + 2) {
                sum += self.at(row, column).norm_sqr();
            }
        }
        sum.sqrt()
    }
}

/// `|re| + |im|`, the norm LAPACK's tests read.
fn norm1(value: Complex64) -> f64 {
    value.re.abs() + value.im.abs()
}

/// The plane rotation that zeroes `g` against `f`: `c` real and `s` complex with
/// `[c, s; −s̄, c] [f; g] = [r; 0]`, with `r` returned too; by LAPACK's ZLARTG (Anderson,
/// "Algorithm 978: Safe scaling in the level 1 BLAS", ACM TOMS 44(1), 2017), which scales
/// to neither overflow nor underflow.
fn rotation(first: Complex64, second: Complex64) -> (f64, Complex64, Complex64) {
    let (safe_min, safe_max) = (f64::MIN_POSITIVE, f64::MIN_POSITIVE.recip());
    let root_min = safe_min.sqrt();
    if second == ZERO {
        return (1.0, ZERO, first);
    }
    let second_size = second.re.abs().max(second.im.abs());
    if first == ZERO {
        let radius = if second.re == 0.0 || second.im == 0.0 {
            second_size
        } else if second_size > root_min && second_size < (safe_max / 2.0).sqrt() {
            second.norm_sqr().sqrt()
        } else {
            let scale = second_size.clamp(safe_min, safe_max);
            (second / scale).norm_sqr().sqrt() * scale
        };
        return (0.0, second.conj() / radius, Complex64::new(radius, 0.0));
    }
    let first_size = first.re.abs().max(first.im.abs());
    let root_max = (safe_max / 4.0).sqrt();
    let unscaled = [first_size, second_size]
        .iter()
        .all(|&size| size > root_min && size < root_max);
    // The pair scaled by `scale`, the first by `scale * ratio`, so their squares neither
    // overflow nor underflow.
    let (scale, ratio) = if unscaled {
        (1.0, 1.0)
    } else {
        let scale = first_size.max(second_size).clamp(safe_min, safe_max);
        if first_size / scale < root_min {
            let first_scale = first_size.clamp(safe_min, safe_max);
            (scale, first_scale / scale)
        } else {
            (scale, 1.0)
        }
    };
    let first_scaled = first / (scale * ratio);
    let second_scaled = second / scale;
    let first_square = first_scaled.norm_sqr();
    let sum = first_square * ratio * ratio + second_scaled.norm_sqr();
    let (cosine, sine, radius) = if first_square >= sum * safe_min {
        let cosine = (first_square / sum).sqrt();
        let radius = first_scaled / cosine;
        let sine = if first_square > root_min && sum < root_max * 2.0 {
            (first_scaled / first_square.sqrt()) * (second_scaled.conj() / sum.sqrt())
        } else {
            second_scaled.conj() * (radius / sum)
        };
        (cosine, sine, radius)
    } else {
        let root = (first_square * sum).sqrt();
        let cosine = first_square / root;
        let radius = if cosine >= safe_min {
            first_scaled / cosine
        } else {
            first_scaled * (sum / root)
        };
        let sine = (first_scaled / first_square.sqrt()) * (second_scaled.conj() / sum.sqrt());
        (cosine, sine, radius)
    };
    (cosine * ratio, sine, radius * scale)
}

/// `x / y` by Smith's algorithm, which neither overflows nor underflows where the quotient
/// does not, as LAPACK's ZLADIV.
fn divide(x: Complex64, y: Complex64) -> Complex64 {
    if y.re.abs() >= y.im.abs() {
        let ratio = y.im / y.re;
        let denominator = y.re + y.im * ratio;
        Complex64::new(
            (x.re + x.im * ratio) / denominator,
            (x.im - x.re * ratio) / denominator,
        )
    } else {
        let ratio = y.re / y.im;
        let denominator = y.im + y.re * ratio;
        Complex64::new(
            (x.re * ratio + x.im) / denominator,
            (x.im * ratio - x.re) / denominator,
        )
    }
}

/// Makes `t` upper triangular by rotations of its rows, applied to `h`'s rows too, which
/// leaves the pencil's eigenvalues where they were: the QR factorization LAPACK's ZGGEV
/// takes before ZGGHRD.
fn triangularize(h: &mut Square, t: &mut Square) {
    let order = h.n;
    for column in 0..order {
        for row in (column + 1..order).rev() {
            let (cosine, sine, radius) = rotation(t.at(row - 1, column), t.at(row, column));
            t.set(row - 1, column, radius);
            t.set(row, column, ZERO);
            t.rotate_rows((row - 1, row), column + 1..order, (cosine, sine));
            h.rotate_rows((row - 1, row), 0..order, (cosine, sine));
        }
    }
}

/// Reduces `h` to upper Hessenberg form and keeps `t` upper triangular, by rotations of rows
/// and of columns, as LAPACK's ZGGHRD.
fn reduce_to_hessenberg(h: &mut Square, t: &mut Square) {
    let order = h.n;
    for column in 0..order.saturating_sub(2) {
        for row in (column + 2..order).rev() {
            // Rotate rows `row - 1` and `row` to zero h's entry in `column`.
            let (cosine, sine, radius) = rotation(h.at(row - 1, column), h.at(row, column));
            h.set(row - 1, column, radius);
            h.set(row, column, ZERO);
            h.rotate_rows((row - 1, row), column + 1..order, (cosine, sine));
            t.rotate_rows((row - 1, row), row - 1..order, (cosine, sine));
            // Rotate columns `row` and `row - 1` to zero the entry it put below t's diagonal.
            let (cosine, sine, radius) = rotation(t.at(row, row), t.at(row, row - 1));
            t.set(row, row, radius);
            t.set(row, row - 1, ZERO);
            h.rotate_columns((row, row - 1), 0..order, (cosine, sine));
            t.rotate_columns((row, row - 1), 0..row, (cosine, sine));
        }
    }
}

/// What the search for where the active block splits found.
enum Split {
    /// The last row's subdiagonal entry is negligible: its eigenvalue is found.
    Found,
    /// The triangular matrix's last diagonal entry is zero: an infinite eigenvalue, split
    /// off by zeroing the Hessenberg matrix's last subdiagonal entry.
    Infinite,
    /// The rows from this one to the last have no negligible subdiagonal entry: a block to
    /// sweep.
    Sweep(usize),
}

/// The tolerances and scales of the QZ iteration, from the norms of the reduced pencil.
#[derive(Clone, Copy)]
struct Scales {
    /// Below these, an entry of the Hessenberg matrix, or a diagonal entry of the
    /// triangular one, is negligible.
    h_tolerance: f64,
    t_tolerance: f64,
    /// The inverses of the two matrices' norms, which the shifts are computed at.
    h_scale: f64,
    t_scale: f64,
}

/// The eigenvalues of the Hessenberg-triangular pencil `(h, t)`, by single-shift QZ sweeps,
/// as LAPACK's ZHGEQZ computes them without the Schur form.
fn qz(h: &mut Square, t: &mut Square) -> Result<Vec<Eigenvalue>, PencilError> {
    let n = h.n;
    let mut eigenvalues = vec![
        Eigenvalue {
            alpha: ZERO,
            beta: ZERO
        };
        n
    ];
    if n == 0 {
        return Ok(eigenvalues);
    }
    let (h_norm, t_norm) = (h.hessenberg_norm(), t.hessenberg_norm());
    let scales = Scales {
        h_tolerance: f64::MIN_POSITIVE.max(f64::EPSILON * h_norm),
        t_tolerance: f64::MIN_POSITIVE.max(f64::EPSILON * t_norm),
        h_scale: f64::MIN_POSITIVE.max(h_norm).recip(),
        t_scale: f64::MIN_POSITIVE.max(t_norm).recip(),
    };
    // The active block's rows and columns run from `first` to `last`; the eigenvalues past
    // `last` are found. `sweeps` counts the sweeps since the last eigenvalue was found, and
    // `exceptional` accumulates the exceptional shifts taken meanwhile.
    let (mut first, mut last) = (0, n - 1);
    let (mut sweeps, mut exceptional) = (0_usize, ZERO);
    for _ in 0..30 * n {
        match split(h, t, (first, last), scales) {
            Split::Sweep(top) => {
                sweeps += 1;
                first = top;
                let shift = shift(h, t, last, (sweeps, &mut exceptional), scales);
                sweep(h, t, (first, last), shift, scales);
                continue;
            }
            Split::Infinite => {
                let (cosine, sine, radius) = rotation(h.at(last, last), h.at(last, last - 1));
                h.set(last, last, radius);
                h.set(last, last - 1, ZERO);
                h.rotate_columns((last, last - 1), first..last, (cosine, sine));
                t.rotate_columns((last, last - 1), first..last, (cosine, sine));
            }
            Split::Found => {}
        }
        // The eigenvalue at `last`, its denominator made real and nonnegative.
        let magnitude = t.at(last, last).norm();
        if magnitude > f64::MIN_POSITIVE {
            let sign = (t.at(last, last) / magnitude).conj();
            t.set(last, last, Complex64::new(magnitude, 0.0));
            h.set(last, last, h.at(last, last) * sign);
        } else {
            t.set(last, last, ZERO);
        }
        eigenvalues[last] = Eigenvalue {
            alpha: h.at(last, last),
            beta: t.at(last, last),
        };
        if last == 0 {
            return Ok(eigenvalues);
        }
        last -= 1;
        sweeps = 0;
        exceptional = ZERO;
        if first > last {
            first = 0;
        }
    }
    Err(PencilError::Unconverged)
}

/// Where the active block of rows `first` to `last` splits, deflating an infinite
/// eigenvalue found within it by rotations that chase its zero out of the block.
fn split(h: &mut Square, t: &mut Square, (first, last): (usize, usize), scales: Scales) -> Split {
    // Whether the subdiagonal entry of `row` is negligible beside its diagonal neighbours.
    let negligible = |h: &Square, row: usize| {
        norm1(h.at(row, row - 1))
            <= f64::MIN_POSITIVE
                .max(f64::EPSILON * (norm1(h.at(row, row)) + norm1(h.at(row - 1, row - 1))))
    };
    if last == 0 {
        return Split::Found;
    }
    if negligible(h, last) {
        h.set(last, last - 1, ZERO);
        return Split::Found;
    }
    if t.at(last, last).norm() <= scales.t_tolerance {
        t.set(last, last, ZERO);
        return Split::Infinite;
    }
    let mut row = last - 1;
    loop {
        let splits = row == 0 || negligible(h, row);
        if splits && row > 0 {
            h.set(row, row - 1, ZERO);
        }
        if t.at(row, row).norm() < scales.t_tolerance {
            t.set(row, row, ZERO);
            return infinite_within(h, t, row, splits, (first, last), scales);
        }
        if splits {
            return Split::Sweep(row);
        }
        row -= 1;
    }
}

/// Deflates the infinite eigenvalue of the zero at `t`'s diagonal entry `row`, within the
/// active block of rows `first` to `last`, where the block `splits` above `row` or not.
fn infinite_within(
    h: &mut Square,
    t: &mut Square,
    row: usize,
    splits: bool,
    (first, last): (usize, usize),
    scales: Scales,
) -> Split {
    // Two consecutive subdiagonal entries small enough together also split the block.
    let mut both_small = !splits
        && norm1(h.at(row, row - 1)) * (scales.h_scale * norm1(h.at(row + 1, row)))
            <= norm1(h.at(row, row)) * (scales.h_scale * scales.h_tolerance);
    if splits || both_small {
        // Split the eigenvalue off at the top of the block, and again while the next
        // diagonal entry of `t` is zero too.
        for k in row..last {
            let (cosine, sine, radius) = rotation(h.at(k, k), h.at(k + 1, k));
            h.set(k, k, radius);
            h.set(k + 1, k, ZERO);
            h.rotate_rows((k, k + 1), k + 1..last + 1, (cosine, sine));
            t.rotate_rows((k, k + 1), k + 1..last + 1, (cosine, sine));
            if both_small {
                h.set(k, k - 1, h.at(k, k - 1) * cosine);
            }
            both_small = false;
            if norm1(t.at(k + 1, k + 1)) >= scales.t_tolerance {
                return if k + 1 >= last {
                    Split::Found
                } else {
                    Split::Sweep(k + 1)
                };
            }
            t.set(k + 1, k + 1, ZERO);
        }
        return Split::Infinite;
    }
    // Chase the zero down to `t`'s last diagonal entry.
    for k in row..last {
        let (cosine, sine, radius) = rotation(t.at(k, k + 1), t.at(k + 1, k + 1));
        t.set(k, k + 1, radius);
        t.set(k + 1, k + 1, ZERO);
        if k + 1 < last {
            t.rotate_rows((k, k + 1), k + 2..last + 1, (cosine, sine));
        }
        h.rotate_rows((k, k + 1), k - 1..last + 1, (cosine, sine));
        let (cosine, sine, radius) = rotation(h.at(k + 1, k), h.at(k + 1, k - 1));
        h.set(k + 1, k, radius);
        h.set(k + 1, k - 1, ZERO);
        h.rotate_columns((k, k - 1), first..k + 1, (cosine, sine));
        t.rotate_columns((k, k - 1), first..k, (cosine, sine));
    }
    Split::Infinite
}

/// The shift of the next sweep of the block ending at row `last`, after `sweeps` sweeps
/// without an eigenvalue: the eigenvalue of the trailing two rows and columns of `h t⁻¹`
/// nearest its last diagonal entry, as Wilkinson's, but every tenth sweep an exceptional
/// one, added to those before it, to break a cycle.
fn shift(
    h: &Square,
    t: &Square,
    last: usize,
    (sweeps, exceptional): (usize, &mut Complex64),
    scales: Scales,
) -> Complex64 {
    let (hs, ts) = (scales.h_scale, scales.t_scale);
    let previous = last - 1;
    if sweeps % 10 != 0 {
        // t = U D, U with a unit diagonal: the trailing block of (h D⁻¹) U⁻¹.
        let u12 = (ts * t.at(previous, last)) / (ts * t.at(last, last));
        let ad11 = (hs * h.at(previous, previous)) / (ts * t.at(previous, previous));
        let ad21 = (hs * h.at(last, previous)) / (ts * t.at(previous, previous));
        let ad12 = (hs * h.at(previous, last)) / (ts * t.at(last, last));
        let ad22 = (hs * h.at(last, last)) / (ts * t.at(last, last));
        let abi22 = ad22 - u12 * ad21;
        let abi12 = ad12 - u12 * ad11;
        let mut shift = abi22;
        let product = abi12.sqrt() * ad21.sqrt();
        if product != ZERO {
            let half = 0.5 * (ad11 - shift);
            let half_size = norm1(half);
            let size = norm1(product).max(half_size);
            let (x, p) = (half / size, product / size);
            let mut root = size * (x * x + p * p).sqrt();
            if half_size > 0.0 {
                let direction = half / half_size;
                if direction.re * root.re + direction.im * root.im < 0.0 {
                    root = -root;
                }
            }
            shift -= product * divide(product, half + root);
        }
        shift
    } else {
        *exceptional += if sweeps % 20 == 0 && ts * norm1(t.at(last, last)) > f64::MIN_POSITIVE {
            (hs * h.at(last, last)) / (ts * t.at(last, last))
        } else {
            (hs * h.at(last, previous)) / (ts * t.at(previous, previous))
        };
        *exceptional
    }
}

/// One implicitly shifted QZ sweep of the block of rows `first` to `last`, by `shift`,
/// started where two consecutive subdiagonal entries are small enough together.
fn sweep(
    h: &mut Square,
    t: &mut Square,
    (first, last): (usize, usize),
    shift: Complex64,
    scales: Scales,
) {
    let (hs, ts) = (scales.h_scale, scales.t_scale);
    let mut start = first;
    let mut lead = hs * h.at(first, first) - shift * (ts * t.at(first, first));
    for row in (first + 1..last).rev() {
        let candidate = hs * h.at(row, row) - shift * (ts * t.at(row, row));
        let (mut diagonal, mut below) = (norm1(candidate), hs * norm1(h.at(row + 1, row)));
        let largest = diagonal.max(below);
        if largest < 1.0 && largest != 0.0 {
            diagonal /= largest;
            below /= largest;
        }
        if norm1(h.at(row, row - 1)) * below <= diagonal * scales.h_tolerance {
            start = row;
            lead = candidate;
            break;
        }
    }
    let (mut cosine, mut sine, _) = rotation(lead, hs * h.at(start + 1, start));
    for row in start..last {
        if row > start {
            let (c, s, radius) = rotation(h.at(row, row - 1), h.at(row + 1, row - 1));
            h.set(row, row - 1, radius);
            h.set(row + 1, row - 1, ZERO);
            (cosine, sine) = (c, s);
        }
        h.rotate_rows((row, row + 1), row..last + 1, (cosine, sine));
        t.rotate_rows((row, row + 1), row..last + 1, (cosine, sine));
        let (c, s, radius) = rotation(t.at(row + 1, row + 1), t.at(row + 1, row));
        t.set(row + 1, row + 1, radius);
        t.set(row + 1, row, ZERO);
        h.rotate_columns((row + 1, row), first..(row + 2).min(last) + 1, (c, s));
        t.rotate_columns((row + 1, row), first..row + 1, (c, s));
    }
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
        assert_eq!(generalized_eigenvalues(0, &[], &[]).expect("empty"), []);
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
    fn subnormal_entries_breed_no_nan() {
        // a = diag(1, 2, 3) but for four subnormal entries, whose squares underflow, which
        // a rotation computed without scaling turns into NaN; b = I. The eigenvalues are
        // found to the rounding of the rotations that zero those entries, a few units in
        // the last place.
        let tiny = 1e-310;
        let a = [1.0, tiny, tiny, 0.0, 2.0, 0.0, tiny, tiny, 3.0];
        let b = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let found = finite(&generalized_eigenvalues(3, &a, &b).expect("converges"));
        let mut eigenvalues: Vec<f64> = found.iter().map(|eigenvalue| eigenvalue.re).collect();
        eigenvalues.sort_by(f64::total_cmp);
        for (eigenvalue, exact) in eigenvalues.iter().zip([1.0, 2.0, 3.0]) {
            assert!(
                (eigenvalue - exact).abs() <= 4.0 * f64::EPSILON * exact,
                "{found:?}"
            );
        }
    }

    #[test]
    fn rejects_non_finite_entries_immediately() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                generalized_eigenvalues(1, &[bad], &[1.0]),
                Err(PencilError::Unconverged)
            );
            assert_eq!(
                generalized_eigenvalues(1, &[1.0], &[bad]),
                Err(PencilError::Unconverged)
            );
            let a = [1.0, 0.0, 0.0, bad];
            let b = [1.0, 0.0, 0.0, 1.0];
            assert_eq!(
                generalized_eigenvalues(2, &a, &b),
                Err(PencilError::Unconverged)
            );
            assert_eq!(
                generalized_eigenvalues(2, &b, &a),
                Err(PencilError::Unconverged)
            );
        }
    }
}
