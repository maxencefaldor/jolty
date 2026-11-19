//! Operation-for-operation `NumPy` and Python arithmetic matching jolty's original Python
//! measures.
//!
//! Each helper mirrors the corresponding `NumPy` 2.5 or Python 3.15 routine so measurements
//! stay bit-identical on platforms with fused multiply-add. Where `NumPy` SIMD loops or
//! arm64 Clang fuse multiply-adds into a single rounding, these helpers use [`f64::mul_add`].

use std::f64::consts::PI;

use jolty_numerics::Complex64;

use super::{Edge, MeasureError, Point, SumError};

/// Local search window around a guess scanned linearly before bisecting (`LIKELY_IN_CACHE_SIZE`
/// in `NumPy`'s `compiled_base.c`).
const LIKELY_IN_CACHE: usize = 8;

/// Interval classification returned by `NumPy`-compatible binary search over ascending points.
enum Found {
    /// Before the first point.
    Below,
    /// Index `j` satisfying `points[j] <= key < points[j + 1]`, or the last index at the upper
    /// endpoint.
    At(usize),
    /// After the last point.
    Above,
}

/// Locates `key` in `points` starting from `guess`, mirroring `NumPy`'s
/// `binary_search_with_guess`.
fn search(key: f64, points: &[f64], guess: usize) -> Found {
    let len = points.len();
    if key > points[len - 1] {
        return Found::Above;
    }
    if key < points[0] {
        return Found::Below;
    }
    if len <= 4 {
        let mut index = 1;
        while index < len && key >= points[index] {
            index += 1;
        }
        return Found::At(index - 1);
    }
    let guess = guess.clamp(1, len - 3);
    let (mut low, mut high) = (0, len);
    if key < points[guess] {
        if key >= points[guess - 1] {
            return Found::At(guess - 1);
        }
        high = guess - 1;
        if guess > LIKELY_IN_CACHE && key >= points[guess - LIKELY_IN_CACHE] {
            low = guess - LIKELY_IN_CACHE;
        }
    } else if key < points[guess + 1] {
        return Found::At(guess);
    } else if key < points[guess + 2] {
        return Found::At(guess + 1);
    } else {
        low = guess + 2;
        if guess + LIKELY_IN_CACHE + 1 < len && key < points[guess + LIKELY_IN_CACHE] {
            high = guess + LIKELY_IN_CACHE;
        }
    }
    while low < high {
        let middle = low + ((high - low) >> 1);
        if key >= points[middle] {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    // The bisection only raises `low` from where it starts, 1 or more, or from 0 past a point
    // the key is not below, so it ends at 1 or more.
    Found::At(low.saturating_sub(1))
}

/// Interpolates `(points, values)` linearly at each coordinate in `at`, mirroring `NumPy`'s
/// `interp` (`arr_interp` in `compiled_base.c`).
///
/// Each lookup resumes from the preceding index so out-of-order inputs match `NumPy` exactly,
/// `NaN` queries propagate unchanged, and interpolation uses a single fused multiply-add
/// rounding. Returns all `NaN` when `points` or `values` is empty.
pub(crate) fn interpolate(at: &[f64], points: &[f64], values: &[f64]) -> Vec<f64> {
    let len = points.len().min(values.len());
    let (points, values) = (&points[..len], &values[..len]);
    let (Some(&left), Some(&right)) = (values.first(), values.last()) else {
        return vec![f64::NAN; at.len()];
    };
    if len == 1 {
        let (point, value) = (points[0], values[0]);
        return at
            .iter()
            .map(|&x| {
                if x < point {
                    left
                } else if x > point {
                    right
                } else {
                    value
                }
            })
            .collect();
    }
    let mut guess = 0;
    at.iter()
        .map(|&x| {
            if x.is_nan() {
                return x;
            }
            match search(x, points, guess) {
                Found::Below => {
                    guess = 0;
                    left
                }
                Found::Above => {
                    guess = len;
                    right
                }
                Found::At(found) => {
                    guess = found;
                    if found == len - 1 || same(points[found], x) {
                        return values[found];
                    }
                    let slope =
                        (values[found + 1] - values[found]) / (points[found + 1] - points[found]);
                    let mut value = slope.mul_add(x - points[found], values[found]);
                    if value.is_nan() {
                        value = slope.mul_add(x - points[found + 1], values[found + 1]);
                        if value.is_nan() && same(values[found], values[found + 1]) {
                            value = values[found];
                        }
                    }
                    value
                }
            }
        })
        .collect()
}

/// Compares `a` and `b` for exact IEEE-754 equality (`==`), matching `NumPy` and Python
/// float equality checks.
#[expect(
    clippy::float_cmp,
    reason = "numpy and Python compare these floats exactly, and so must the functions that \
              follow them"
)]
pub(crate) fn same(a: f64, b: f64) -> bool {
    a == b
}

/// Computes `|z|` matching `NumPy`'s `simd_cabsolute` loop using fused multiply-add,
/// returning infinity if either part is infinite and `NaN` if either is `NaN`.
pub(crate) fn magnitude(z: Complex64) -> f64 {
    let (mut real, mut imaginary) = (z.re.abs(), z.im.abs());
    let (real_infinite, imaginary_infinite) = (real == f64::INFINITY, imaginary == f64::INFINITY);
    if real_infinite {
        imaginary = f64::INFINITY;
    }
    if imaginary_infinite {
        real = f64::INFINITY;
    }
    let (real_nan, imaginary_nan) = (real.is_nan(), imaginary.is_nan());
    if real_nan {
        imaginary = f64::NAN;
    }
    if imaginary_nan {
        real = f64::NAN;
    }
    let larger = real.max(imaginary);
    let smaller = imaginary.min(real);
    let ratio = if larger == 0.0 || smaller == f64::INFINITY {
        0.0
    } else {
        smaller / larger
    };
    ratio.mul_add(ratio, 1.0).sqrt() * larger
}

/// Computes complex quotient `a / b` via Smith's algorithm matching `CDOUBLE_divide` in
/// `NumPy` with fused multiply-add rounding.
pub(crate) fn divide(a: Complex64, b: Complex64) -> Complex64 {
    let (real, imaginary) = (b.re.abs(), b.im.abs());
    if real >= imaginary {
        if real == 0.0 && imaginary == 0.0 {
            return Complex64::new(a.re / real, a.im / imaginary);
        }
        let ratio = b.im / b.re;
        let scale = 1.0 / b.im.mul_add(ratio, b.re);
        Complex64::new(
            a.im.mul_add(ratio, a.re) * scale,
            (-a.re).mul_add(ratio, a.im) * scale,
        )
    } else {
        let ratio = b.re / b.im;
        let scale = 1.0 / b.re.mul_add(ratio, b.im);
        Complex64::new(
            a.re.mul_add(ratio, a.im) * scale,
            a.im.mul_add(ratio, -a.re) * scale,
        )
    }
}

/// Computes `20 * log10(|z|)` in decibels matching `NumPy`, returning `-inf` at zero.
pub(crate) fn decibels(z: Complex64) -> f64 {
    20.0 * magnitude(z).log10()
}

/// Computes unwrapped phases of `transfer` in degrees, matching `NumPy`'s
/// `degrees(unwrap(angle(transfer)))`.
pub(crate) fn phases(transfer: &[Complex64]) -> Vec<f64> {
    let angles: Vec<f64> = transfer.iter().map(|z| z.im.atan2(z.re)).collect();
    unwrap(&angles)
        .into_iter()
        .map(|angle| angle * (180.0 / PI))
        .collect()
}

/// Unwraps radian `angles` with period `2π` matching `NumPy`'s `unwrap`, folding steps wider
/// than `π` back into `[-π, π]` and accumulating corrections from the first sample.
pub(crate) fn unwrap(angles: &[f64]) -> Vec<f64> {
    let period = 2.0 * PI;
    let high = period / 2.0;
    let low = -high;
    let Some(&first) = angles.first() else {
        return Vec::new();
    };
    let mut unwrapped = Vec::with_capacity(angles.len());
    unwrapped.push(first);
    let mut total: Option<f64> = None;
    for pair in angles.windows(2) {
        let step = pair[1] - pair[0];
        let mut wrapped = remainder(step - low, period) + low;
        if same(wrapped, low) && step > 0.0 {
            wrapped = high;
        }
        let correction = if step.abs() < high {
            0.0
        } else {
            wrapped - step
        };
        // `NumPy` starts cumulative sum from the first correction rather than zero.
        let sum = total.map_or(correction, |sum| sum + correction);
        total = Some(sum);
        unwrapped.push(pair[1] + sum);
    }
    unwrapped
}

/// Computes floored remainder `a mod b` carrying the sign of `b`, matching `npy_remainder`
/// in `NumPy` and `%` in Python.
fn remainder(a: f64, b: f64) -> f64 {
    let modulus = a % b;
    if b == 0.0 {
        return modulus;
    }
    if modulus == 0.0 {
        0.0_f64.copysign(b)
    } else if (b < 0.0) == (modulus < 0.0) {
        modulus
    } else {
        modulus + b
    }
}

/// Sums `count` terms produced by `term` in `NumPy`'s `pairwise_sum` order
/// (`loops_utils.h.src`), adding `0.0` at the end to normalize `-0.0`.
pub(crate) fn sum(count: usize, term: impl Fn(usize) -> f64) -> f64 {
    0.0 + pairwise(0, count, &term)
}

/// Block size summed with eight lanes in `NumPy`'s pairwise reduction (`PW_BLOCKSIZE` in
/// `loops_utils.h.src`).
const BLOCK: usize = 128;

/// Reduces `count` terms from `start` recursively using `NumPy`'s eight-accumulator pairwise
/// block summation up to [`BLOCK`].
fn pairwise(start: usize, count: usize, term: &impl Fn(usize) -> f64) -> f64 {
    if count < 8 {
        // `NumPy` starts at `-0.0` so summing all `-0.0` preserves negative zero within a block.
        let mut total = -0.0;
        for index in start..start + count {
            total += term(index);
        }
        total
    } else if count <= BLOCK {
        let mut lanes = [0.0; 8];
        for (lane, value) in lanes.iter_mut().enumerate() {
            *value = term(start + lane);
        }
        let whole = count - count % 8;
        let mut index = 8;
        while index < whole {
            for (lane, value) in lanes.iter_mut().enumerate() {
                *value += term(start + index + lane);
            }
            index += 8;
        }
        let mut total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
        while index < count {
            total += term(start + index);
            index += 1;
        }
        total
    } else {
        let half = count / 2;
        let half = half - half % 8;
        pairwise(start, half, term) + pairwise(start + half, count - half, term)
    }
}

/// Computes a correctly rounded sum over `values` via Shewchuk's exact partial-sum algorithm,
/// matching Python's `math.fsum` (`math_fsum`).
///
/// # Errors
///
/// Returns [`SumError::Overflow`] if an intermediate finite sum overflows, or
/// [`SumError::Indeterminate`] if `values` contains both `+inf` and `-inf`.
pub(crate) fn exact_sum(values: impl IntoIterator<Item = f64>) -> Result<f64, SumError> {
    let mut partials: Vec<f64> = Vec::new();
    let (mut special, mut infinite) = (0.0, 0.0);
    for value in values {
        let mut x = value;
        let mut kept = 0;
        for index in 0..partials.len() {
            let mut y = partials[index];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let high = x + y;
            let low = y - (high - x);
            if low != 0.0 {
                partials[kept] = low;
                kept += 1;
            }
            x = high;
        }
        partials.truncate(kept);
        if x != 0.0 {
            if x.is_finite() {
                partials.push(x);
            } else {
                if value.is_finite() {
                    return Err(SumError::Overflow);
                }
                if value.is_infinite() {
                    infinite += value;
                }
                special += value;
                partials.clear();
            }
        }
    }
    if special != 0.0 {
        return if infinite.is_nan() {
            Err(SumError::Indeterminate)
        } else {
            Ok(special)
        };
    }
    let Some(mut high) = partials.pop() else {
        return Ok(0.0);
    };
    let mut low = 0.0;
    while let Some(y) = partials.pop() {
        let x = high;
        high = x + y;
        low = y - (high - x);
        if low != 0.0 {
            break;
        }
    }
    // Rounds half to even across the partials: a remainder of the same sign as the next
    // partial pushes a tie the way the exact sum lies.
    if let Some(&next) = partials.last()
        && ((low < 0.0 && next < 0.0) || (low > 0.0 && next > 0.0))
    {
        let twice = low * 2.0;
        let rounded = high + twice;
        if same(twice, rounded - high) {
            high = rounded;
        }
    }
    Ok(high)
}

/// Returns the maximum of `values` matching `NumPy`'s `max`, propagating `NaN` and preferring
/// `+0.0` over `-0.0` (or `NaN` when empty).
pub(crate) fn maximum(values: &[f64]) -> f64 {
    extreme(values, |value, best| {
        value > best || (same(value, best) && best.is_sign_negative())
    })
}

/// Returns the minimum of `values` matching `NumPy`'s `min`, propagating `NaN` and preferring
/// `-0.0` over `+0.0` (or `NaN` when empty).
pub(crate) fn minimum(values: &[f64]) -> f64 {
    extreme(values, |value, best| {
        value < best || (same(value, best) && value.is_sign_negative())
    })
}

/// Reduces `values` under comparator `beats`, propagating `NaN` immediately.
fn extreme(values: &[f64], beats: impl Fn(f64, f64) -> bool) -> f64 {
    let mut best = f64::NAN;
    for (index, &value) in values.iter().enumerate() {
        if value.is_nan() {
            return value;
        }
        if index == 0 || beats(value, best) {
            best = value;
        }
    }
    best
}

/// Integrates `values` over `points` using the trapezoidal rule in `NumPy`'s pairwise order,
/// matching `NumPy`'s `trapezoid`.
pub(crate) fn trapezoid(points: &[f64], values: &[f64]) -> f64 {
    let count = points.len().min(values.len()).saturating_sub(1);
    sum(count, |index| {
        (points[index + 1] - points[index]) * (values[index + 1] + values[index]) / 2.0
    })
}

/// Yields linearly interpolated coordinates where `(points, values)` crosses `threshold` in
/// direction `edge`, matching jolty's Python `_crossings`.
pub(crate) fn crossings<'a>(
    points: &'a [f64],
    values: &'a [f64],
    threshold: f64,
    edge: Edge,
) -> impl Iterator<Item = f64> + 'a {
    points
        .windows(2)
        .zip(values.windows(2))
        .filter_map(move |(around, pair)| {
            let (below, above) = (pair[0] - threshold, pair[1] - threshold);
            let rising = below < 0.0 && above >= 0.0;
            let falling = below > 0.0 && above <= 0.0;
            let crossed = match edge {
                Edge::Rising => rising,
                Edge::Falling => falling,
                Edge::Either => rising || falling,
            };
            crossed.then(|| {
                let fraction = below / (below - above);
                around[0] + fraction * (around[1] - around[0])
            })
        })
}

/// Computes ascending sort indices for `values`, placing `NaN` last and preserving relative
/// order on ties, matching `NumPy`'s `argsort`.
pub(crate) fn order(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        let (a, b) = (values[a], values[b]);
        a.partial_cmp(&b)
            .unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
    });
    order
}

/// Sorts `frequencies` ascending and reorders `densities` to match.
///
/// # Errors
///
/// Returns [`MeasureError::Empty`] for empty inputs and [`MeasureError::LengthMismatch`] when
/// lengths differ.
fn sorted(frequencies: &[f64], densities: &[f64]) -> Result<(Vec<f64>, Vec<f64>), MeasureError> {
    if frequencies.len() != densities.len() {
        return Err(MeasureError::LengthMismatch);
    }
    if frequencies.is_empty() {
        return Err(MeasureError::Empty);
    }
    let order = order(frequencies);
    Ok((
        order.iter().map(|&index| frequencies[index]).collect(),
        order.iter().map(|&index| densities[index]).collect(),
    ))
}

/// Evaluates power-law interpolated noise power density at frequency `at` over
/// `(frequencies, densities)`.
///
/// # Errors
///
/// Returns [`MeasureError::Outside`] when `at` falls outside the frequency range,
/// [`MeasureError::Empty`] for empty inputs, and [`MeasureError::LengthMismatch`] when
/// lengths differ.
pub fn density(frequencies: &[f64], densities: &[f64], at: f64) -> Result<f64, MeasureError> {
    let (points, values) = sorted(frequencies, densities)?;
    let (first, last) = (points[0], points[points.len() - 1]);
    if !(first <= at && at <= last) {
        return Err(MeasureError::Outside {
            point: Point::At,
            first,
            last,
        });
    }
    Ok(power_law(&points, &values, at))
}

/// Integrates piecewise power-law noise `densities` over `frequencies` from `start` to `stop`
/// (defaulting to the full range), matching SPICE noise integration.
///
/// # Errors
///
/// Returns [`MeasureError::Band`] for invalid or out-of-range bands, [`MeasureError::Sum`]
/// on exact summation overflow or indeterminate infinity, [`MeasureError::Empty`] for empty
/// inputs, and [`MeasureError::LengthMismatch`] when lengths differ.
pub fn integrate(
    frequencies: &[f64],
    densities: &[f64],
    start: Option<f64>,
    stop: Option<f64>,
) -> Result<f64, MeasureError> {
    let (points, values) = sorted(frequencies, densities)?;
    let (first, last) = (points[0], points[points.len() - 1]);
    let start = start.unwrap_or(first);
    let stop = stop.unwrap_or(last);
    if !(first <= start && start < stop && stop <= last) {
        return Err(MeasureError::Band {
            start,
            stop,
            first,
            last,
        });
    }
    let inside = (0..points.len()).filter(|&index| points[index] > start && points[index] < stop);
    let mut band = vec![start];
    let mut densities_in_band = vec![power_law(&points, &values, start)];
    for index in inside {
        band.push(points[index]);
        densities_in_band.push(values[index]);
    }
    band.push(stop);
    densities_in_band.push(power_law(&points, &values, stop));
    exact_sum(
        band.windows(2)
            .zip(densities_in_band.windows(2))
            .map(|(ends, densities)| segment(ends[0], ends[1], densities[0], densities[1])),
    )
    .map_err(MeasureError::Sum)
}

/// Interpolates density at `at` between adjacent ascending `points` using power-law scaling
/// (falling back to linear interpolation when either endpoint is non-positive).
fn power_law(points: &[f64], values: &[f64], at: f64) -> f64 {
    let count = points.len();
    // Clip index to `[1, count - 1]` (yielding 0 for a single point), matching `NumPy`.
    let index = points
        .partition_point(|&point| point < at)
        .max(1)
        .min(count - 1);
    let before = index.checked_sub(1).unwrap_or(count - 1);
    let (first, last) = (points[before], points[index]);
    let (low, high) = (values[before], values[index]);
    if same(at, first) {
        return low;
    }
    if same(at, last) {
        return high;
    }
    if low <= 0.0 || high <= 0.0 {
        return low + (high - low) * (at - first) / (last - first);
    }
    low * ((high / low).ln() * (at / first).ln() / (last / first).ln()).exp()
}

/// Integrates a single power-law segment from `(first, low)` to `(last, high)`, falling back
/// to trapezoidal area when either endpoint density is non-positive.
fn segment(first: f64, last: f64, low: f64, high: f64) -> f64 {
    if low <= 0.0 || high <= 0.0 {
        #[expect(
            clippy::manual_midpoint,
            reason = "jolty's `0.5 * (low + high)`, which overflows where `f64::midpoint` would not"
        )]
        let mean = 0.5 * (low + high);
        return mean * (last - first);
    }
    let span = (last / first).ln();
    // With density proportional to `f^a`, the integral is `low * first * (e^{(a+1) * span} - 1) / (a + 1)`.
    let growth = (high / low).ln() + span;
    let factor = if growth == 0.0 {
        1.0
    } else {
        growth.exp_m1() / growth
    };
    low * first * span * factor
}

/// Computes input-referred noise densities `density / |gain|^2` over `densities` and `gains`,
/// producing infinity where gain vanishes.
#[must_use]
pub fn referred(densities: &[f64], gains: &[Complex64]) -> Vec<f64> {
    densities
        .iter()
        .zip(gains)
        .map(|(density, &gain)| {
            let magnitude = magnitude(gain);
            density / (magnitude * magnitude)
        })
        .collect()
}

/// Computes damping ratios `-Re(s) / |s|` for `roots` (`1` for negative real roots, `(0, 1)`
/// for damped conjugate pairs, negative in the right half-plane, and `NaN` at zero).
#[must_use]
pub fn dampings(roots: &[Complex64]) -> Vec<f64> {
    roots
        .iter()
        .map(|&root| -root.re / magnitude(root))
        .collect()
}

/// Computes natural frequencies `|s| / (2π)` in hertz for each root in `roots`.
#[must_use]
pub fn natural_frequencies(roots: &[Complex64]) -> Vec<f64> {
    roots
        .iter()
        .map(|&root| magnitude(root) / (2.0 * PI))
        .collect()
}

#[cfg(test)]
mod tests {
    //! Each expected value is what numpy 2.5.4 or Python 3.15's `math` computes for the same
    //! input on an Apple M2 Max, printed by `repr`, which round-trips every bit.

    use super::*;

    #[test]
    fn sums_in_numpys_pairwise_order() {
        let count = 1001;
        let harmonic = |index: usize| 1.0 / f64::from(u32::try_from(index + 1).unwrap());
        // `(1.0 / np.arange(1, 1002)).sum()`; a sum in order gives 7.486469861549344, and
        // `math.fsum` 7.486469861549346.
        assert_eq!(
            sum(count, harmonic).to_bits(),
            7.486_469_861_549_347_f64.to_bits()
        );
        // numpy keeps the sign of a sum of minus zeros only within a block, then adds zero.
        assert!(sum(3, |_| -0.0).is_sign_positive());
    }

    #[test]
    fn sums_exactly_as_math_fsum_does() {
        assert_eq!(
            exact_sum([1e-16, 1.0, 1e16]),
            Ok(1.000_000_000_000_000_2e16)
        );
        assert_eq!(exact_sum([0.1; 10]), Ok(1.0));
        assert_eq!(exact_sum([]), Ok(0.0));
        assert_eq!(exact_sum([1e308, 1e308]), Err(SumError::Overflow));
        assert_eq!(
            exact_sum([f64::INFINITY, -f64::INFINITY]),
            Err(SumError::Indeterminate)
        );
        assert!(exact_sum([1.0, f64::NAN]).unwrap().is_nan());
    }

    #[test]
    fn interpolates_as_numpy_does_even_out_of_order() {
        // `np.interp(x, xp, fp)` over points out of order, searched from each last one found.
        let points = [0.0, 1.0, 0.5, 2.0, 3.0, 1.5, 4.0];
        let values = [0.0, 10.0, -5.0, 3.0, 7.0, 1.0, 2.0];
        let at = [0.25, 1.75, 3.5, 0.75, 2.5, 5.0, -1.0, 1.2];
        let expected: [f64; 8] = [
            2.5,
            1.666_666_666_666_666_3,
            1.8,
            7.5,
            1.4,
            2.0,
            0.0,
            -1.266_666_666_666_667,
        ];
        let found = interpolate(&at, &points, &values);
        assert_eq!(
            found
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        // One rounding, as numpy's on arm64: two would give 0.44285714285714284.
        assert_eq!(
            interpolate(&[0.3], &[0.0, 0.7], &[0.1, 0.9]),
            [0.442_857_142_857_142_9]
        );
        assert!(
            interpolate(&[0.5], &[], &[])
                .iter()
                .all(|value| value.is_nan())
        );
    }

    #[test]
    fn magnitudes_and_phases_are_numpys() {
        let magnitudes = [
            Complex64::new(3.0, 4.0),
            Complex64::new(1e-300, 1e-300),
            Complex64::new(-2.5, 0.1),
            Complex64::new(0.0, 0.0),
        ]
        .map(magnitude);
        assert_eq!(
            magnitudes,
            [
                5.0,
                1.414_213_562_373_095_2e-300,
                2.501_999_200_639_360_8,
                0.0
            ]
        );
        assert_eq!(
            magnitude(Complex64::new(f64::INFINITY, f64::NAN)),
            f64::INFINITY
        );
        assert!(magnitude(Complex64::new(1.0, f64::NAN)).is_nan());
        let unwrapped = unwrap(&[0.0, 3.0, -3.0, 0.5, 6.5, -6.0]);
        assert_eq!(
            unwrapped,
            [
                0.0,
                3.0,
                3.283_185_307_179_586_2,
                0.5,
                0.216_814_692_820_413_77,
                0.283_185_307_179_586_23
            ]
        );
    }

    #[test]
    fn integrates_by_the_trapezoidal_rule_in_numpys_order() {
        let points: Vec<f64> = (0..200).map(|index| f64::from(index) * 0.05).collect();
        let values: Vec<f64> = (0..200)
            .map(|index| (f64::from(index) * 0.1).sin())
            .collect();
        // `np.trapezoid(np.sin(np.arange(200) * 0.1), np.arange(200) * 0.05)`.
        assert_eq!(trapezoid(&points, &values), 0.251_197_561_719_626_15);
    }

    #[test]
    fn extremes_propagate_what_is_not_a_number() {
        assert!(maximum(&[1.0, f64::NAN, 2.0]).is_nan());
        assert!(minimum(&[f64::NAN, 1.0]).is_nan());
        assert!(maximum(&[-0.0, 0.0]).is_sign_positive());
        assert!(minimum(&[0.0, -0.0]).is_sign_negative());
        assert!(maximum(&[]).is_nan());
    }
}
